import type { Graph } from './Graph'
import type { Evidence, EdgeType } from './types'

// ====================================================================
// Loopy Belief Propagation — Enhanced Noisy-OR-NOT-AND model
//
// MATHEMATICAL FOUNDATION
// ═══════════════════════
//
// Each node v is a binary variable: P(v = true) ∈ [0, 1]
//
// Node parameters:
//   prior(v)         Base rate in target population              [0,1]
//   sensitivity(v)   How much incoming evidence can move v       [0,1]
//                    0 = immune (stays at prior), 1 = fully responsive
//   sigma(v)         Epistemic uncertainty in the prior          [0,1]
//                    High sigma → more room for evidence to move belief
//   gateType(v)      Activation gate: 'or' | 'and' | 'mixed'    default='or'
//   gateWeight(v)    For mixed gate: 0=AND, 1=OR                 default=0.5
//
// ── GATE TYPES ─────────────────────────────────────────────────────
//
//   OR gate  (Noisy-OR, Pearl 1988) — default:
//     Any parent can activate v independently.
//     posProduct = ∏_i (1 - q_i × P(parent_i))
//     belief_or = 1 - (1 - leak) × posProduct
//
//   AND gate (Noisy-AND) — for nodes requiring multiple conditions:
//     All parents must co-activate v.
//     Uses geometric mean to normalize across parent count:
//       andMean = (∏_i (q_i × P(parent_i)))^(1/n)
//     belief_and = leak + (1 - leak) × andMean
//     If any parent ≈ 0 → andMean → 0 → belief_and ≈ leak ≈ prior
//
//   Mixed gate: belief = w × belief_or + (1-w) × belief_and
//     where w = gateWeight ∈ [0,1]
//
// ── SYNERGY (co-activation non-linearity) ──────────────────────────
//
//   When multiple parents are simultaneously active above SYNERGY_THRESHOLD,
//   the child activates more strongly than linear Noisy-OR predicts.
//   This captures the empirical fact that co-occurrence of risk factors
//   creates super-additive effects.
//
//   Active parents: count = |{i : P(parent_i) > SYNERGY_THRESHOLD}|
//   Synergy exponent: exp = 1 + SYNERGY_BOOST × min(count - 1, 3)
//   Effective product: posProduct_synergy = posProduct^exp
//     (exp > 1 drives posProduct closer to 0 → more activation)
//
// ── NEGATIVE EVIDENCE ──────────────────────────────────────────────
//
//   When evidence.value === false (observed absent):
//     belief_clamped = prior × (1 - confidence)
//   This pushes belief well below the prior, correctly modeling
//   the information gain from a confirmed absence.
//
// ── INHIBITION (weakens) ───────────────────────────────────────────
//
//   negProduct = ∏_j (1 - SCALE_NEG × weight_j × P(inhibitor_j))
//   raw_belief = activated × negProduct
//
// ── DAMPING (anti-oscillation) ─────────────────────────────────────
//
//   Loopy BP on cyclic graphs can oscillate. Damping mixes the new
//   estimate with the previous iteration value:
//     belief_t = DAMPING × computed_t + (1 - DAMPING) × belief_{t-1}
//
// ── SENSITIVITY MODULATION ─────────────────────────────────────────
//
//   belief(v) = prior + sensitivity × (raw_belief - prior)
//   Sensitivity=0 → immune, Sensitivity=1 → fully responsive.
//   Reference anchor = true prior (not sigma-adjusted leak).
//
// ── CONVERGENCE ────────────────────────────────────────────────────
//
//   max |belief_t - belief_{t-1}| < threshold over all nodes
//   Evidence nodes are clamped throughout (never updated).
//
// ====================================================================

// ── Activation scale per edge type ─────────────────────────────────
const TYPE_SCALE_POS: Record<EdgeType, number> = {
  determines: 1.00,
  implies:    1.00,
  enables:    0.95,
  exploits:   0.90,
  reveals:    0.82,
  influences: 0.62,
  suggests:   0.50,
  weakens:    0.00,
}

// Inhibitory scale for 'weakens' edges
const SCALE_NEG_WEAKENS = 0.78

// Damping: mix new estimate with previous to prevent oscillation
// DAMPING=0.80 → 80% new, 20% old; higher = faster but more oscillation risk
const DAMPING = 0.80

// Synergy: co-activation of ≥2 parents amplifies child belief
// Threshold for counting a parent as "firing"
const SYNERGY_THRESHOLD = 0.45
// Boost exponent increment per extra co-active parent (capped at 3 extra)
const SYNERGY_BOOST = 0.28

const DEFAULT_PRIOR       = 0.20
const DEFAULT_SENSITIVITY = 0.85
const DEFAULT_SIGMA       = 0.30
const EPSILON             = 1e-9

function clamp01(x: number): number {
  return x < EPSILON ? 0 : x > 1 - EPSILON ? 1 : x
}

export interface BeliefMap {
  beliefs: Map<string, number>   // nodeId → posterior probability [0, 1]
  iterations: number
  converged: boolean
  maxDelta: number               // largest belief shift in final iteration (diagnostic)
}

export class BeliefPropagation {
  constructor(
    private readonly maxIter: number  = 40,
    private readonly threshold: number = 0.0006
  ) {}

  /**
   * @param observedNodeIds  Canvas node IDs. Only these start at their prior;
   *                         all other nodes start at 0 (unknown).
   */
  propagate(graph: Graph, evidence: Evidence[], observedNodeIds?: Set<string>): BeliefMap {
    const nodes = graph.getAllNodes()
    if (nodes.length === 0) {
      return { beliefs: new Map(), iterations: 0, converged: true, maxDelta: 0 }
    }

    if (observedNodeIds && observedNodeIds.size === 0 && evidence.length === 0) {
      return { beliefs: new Map(), iterations: 0, converged: true, maxDelta: 0 }
    }

    // ── Evidence partitioning ────────────────────────────────────────
    // Positive evidence: value=true (or legacy undefined) → clamp to confidence
    // Negative evidence: value=false → suppress below prior
    const posEvidenceMap = new Map<string, number>()  // nodeId → confidence
    const negEvidenceMap = new Map<string, number>()  // nodeId → confidence of absence

    for (const e of evidence) {
      if (e.value === false) {
        negEvidenceMap.set(e.nodeId, clamp01(e.confidence))
      } else {
        posEvidenceMap.set(e.nodeId, clamp01(e.confidence))
      }
    }

    // ── Pre-compute per-node parameters ─────────────────────────────
    const truePriors = new Map<string, number>()
    const leaks      = new Map<string, number>()
    const sensMap    = new Map<string, number>()

    for (const node of nodes) {
      const p    = clamp01(node.prior       ?? DEFAULT_PRIOR)
      const sig  = clamp01(node.sigma       ?? DEFAULT_SIGMA)
      const sens = clamp01(node.sensitivity ?? DEFAULT_SENSITIVITY)

      truePriors.set(node.id, p)
      sensMap.set(node.id, sens)
      // sigma pulls leak toward 0.5, widening the evidence response range
      leaks.set(node.id, clamp01(p + sig * 0.4 * (0.5 - p)))
    }

    // ── Initialize beliefs ───────────────────────────────────────────
    const beliefs = new Map<string, number>()
    for (const node of nodes) {
      const isObserved = !observedNodeIds || observedNodeIds.has(node.id)
      beliefs.set(node.id, isObserved ? truePriors.get(node.id)! : 0)
    }

    // Clamp positive evidence nodes
    for (const [nodeId, conf] of posEvidenceMap) {
      beliefs.set(nodeId, conf)
    }
    // Clamp negative evidence nodes (suppress below prior)
    for (const [nodeId, conf] of negEvidenceMap) {
      const prior = truePriors.get(nodeId) ?? DEFAULT_PRIOR
      beliefs.set(nodeId, clamp01(prior * (1 - conf)))
    }

    let iterations = 0
    let converged  = false
    let maxDelta   = 0

    for (let iter = 0; iter < this.maxIter; iter++) {
      iterations++
      const prev = new Map(beliefs)
      maxDelta = 0

      for (const node of nodes) {
        // Evidence nodes are clamped — never updated
        if (posEvidenceMap.has(node.id) || negEvidenceMap.has(node.id)) continue

        const isObserved  = !observedNodeIds || observedNodeIds.has(node.id)
        const prior       = isObserved ? truePriors.get(node.id)! : 0
        const leak        = isObserved ? leaks.get(node.id)!      : 0
        const sensitivity = sensMap.get(node.id)!
        const gateType    = node.gateType  ?? 'or'
        const gateWeight  = node.gateWeight ?? 0.5
        const incoming    = graph.getIncomingEdges(node.id)

        if (incoming.length === 0) {
          beliefs.set(node.id, prior)
          continue
        }

        // Partition edges
        const posEdges = incoming.filter(e => e.type !== 'weakens')
        const negEdges = incoming.filter(e => e.type === 'weakens')

        // ── Synergy: count co-active parents ─────────────────────────
        let activeParentCount = 0
        for (const edge of posEdges) {
          if ((prev.get(edge.source) ?? 0) > SYNERGY_THRESHOLD) activeParentCount++
        }
        const synergyExponent = activeParentCount >= 2
          ? 1 + SYNERGY_BOOST * Math.min(activeParentCount - 1, 3)
          : 1.0

        // ── OR gate: Noisy-OR with synergy ────────────────────────────
        let posProduct = 1.0
        for (const edge of posEdges) {
          const srcBelief = prev.get(edge.source) ?? 0
          const q  = TYPE_SCALE_POS[edge.type] ?? 0.50
          const qi = clamp01(q * edge.weight * srcBelief)
          posProduct *= (1 - qi)
        }
        // Apply synergy: exponent > 1 drives posProduct toward 0 (more activation)
        const posProductSyn = posEdges.length > 0 ? Math.pow(posProduct, synergyExponent) : 1.0
        const beliefOr = clamp01(1 - (1 - leak) * posProductSyn)

        // ── AND gate: geometric mean of activations ───────────────────
        // Normalizes for parent count — each parent must be active.
        // If any parent ≈ 0 → geometric mean → 0 → belief_and ≈ leak (prior).
        let beliefAnd = leak
        if (posEdges.length > 0) {
          let logSum = 0
          for (const edge of posEdges) {
            const srcBelief = prev.get(edge.source) ?? 0
            const q = TYPE_SCALE_POS[edge.type] ?? 0.50
            const activation = q * edge.weight * srcBelief
            logSum += Math.log(Math.max(EPSILON, activation))
          }
          const andMean = Math.exp(logSum / posEdges.length)  // geometric mean
          beliefAnd = clamp01(leak + (1 - leak) * andMean)
        }

        // ── Combine gates ─────────────────────────────────────────────
        let activated: number
        if (gateType === 'and') {
          activated = beliefAnd
        } else if (gateType === 'mixed') {
          // gateWeight=1 → pure OR, gateWeight=0 → pure AND
          activated = clamp01(gateWeight * beliefOr + (1 - gateWeight) * beliefAnd)
        } else {
          // default: 'or'
          activated = beliefOr
        }

        // ── Inhibition (weakens → Noisy-AND-NOT) ─────────────────────
        let negProduct = 1.0
        for (const edge of negEdges) {
          const srcBelief = prev.get(edge.source) ?? 0
          const inh = clamp01(SCALE_NEG_WEAKENS * edge.weight * srcBelief)
          negProduct *= (1 - inh)
        }

        const rawBelief = clamp01(activated * negProduct)

        // ── Sensitivity modulation ────────────────────────────────────
        const computed = clamp01(prior + sensitivity * (rawBelief - prior))

        // ── Damping: blend new estimate with previous ─────────────────
        const prevBelief = prev.get(node.id) ?? 0
        const newBelief  = clamp01(DAMPING * computed + (1 - DAMPING) * prevBelief)

        beliefs.set(node.id, newBelief)
      }

      // ── Re-clamp evidence after each iteration ─────────────────────
      for (const [nodeId, conf] of posEvidenceMap) {
        beliefs.set(nodeId, conf)
      }
      for (const [nodeId, conf] of negEvidenceMap) {
        const prior = truePriors.get(nodeId) ?? DEFAULT_PRIOR
        beliefs.set(nodeId, clamp01(prior * (1 - conf)))
      }

      // ── Convergence check ─────────────────────────────────────────────
      for (const [id, val] of beliefs) {
        const delta = Math.abs(val - (prev.get(id) ?? 0))
        if (delta > maxDelta) maxDelta = delta
      }
      if (maxDelta < this.threshold) {
        converged = true
        break
      }
    }

    return { beliefs, iterations, converged, maxDelta }
  }
}

// ── Utilities exposed for calibration and scoring ──────────────────

// Log-odds (logit) transform: maps [0,1] → (-∞, +∞)
// Useful for combining independent evidence additively
export function logit(p: number): number {
  const c = Math.max(EPSILON, Math.min(1 - EPSILON, p))
  return Math.log(c / (1 - c))
}

// Inverse logit (sigmoid): maps (-∞, +∞) → [0,1]
// Numerically stable implementation
export function sigmoid(lo: number): number {
  if (lo >= 0) {
    const e = Math.exp(-lo)
    return 1 / (1 + e)
  }
  const e = Math.exp(lo)
  return e / (1 + e)
}

// Jeffreys interval: credible interval for a binomial proportion
// Returns [lo, hi] 95% credible interval given n observations with k successes
export function jeffreysInterval(k: number, n: number): [number, number] {
  if (n === 0) return [0, 1]
  // Beta(k + 0.5, n - k + 0.5) quantiles — approximated via normal
  const p = (k + 0.5) / (n + 1)
  const se = Math.sqrt(p * (1 - p) / (n + 1))
  return [clamp01(p - 1.96 * se), clamp01(p + 1.96 * se)]
}
