import type { Graph } from '../graph/Graph'
import type { Evidence } from '../types'
import type { RigidityState } from './RigidityModel'
import { computeDeformationCost, computeEdgeConductivity } from './RigidityModel'
import type { AffinityMatrix } from './SemanticAffinity'
import type { TopologyResult } from './BeliefTopology'

// ====================================================================
// EIDOLON — Loopy Belief Propagation Engine
//
// MATHEMATICAL FOUNDATION
// ═══════════════════════
//
// Each node v is a binary variable: P(v = true) ∈ [0, 1]
//
// Node parameters:
//   prior(v)         Base rate in target population              [0,1]
//   sensitivity(v)   Signal amplification (0=immune, 1=responsive) [0,1]
//   sigma(v)         Epistemic uncertainty in the prior           [0,1]
//   gateType(v)      Activation gate: 'or' | 'and' | 'mixed'
//   rigidity(v)      Resistance to external pressure (Phase 2)    [0,1]
//                    ORTHOGONAL to prior — two independent axes
//
// ── GATE TYPES ─────────────────────────────────────────────────────
//
//   OR gate  (Noisy-OR, Pearl 1988) — default:
//     posProduct = ∏_i (1 - q_i × P(parent_i))
//     belief_or = 1 - (1 - leak) × posProduct
//
//   AND gate (Noisy-AND):
//     andMean = (∏_i (q_i × P(parent_i)))^(1/n)
//     belief_and = leak + (1 - leak) × andMean
//
//   Mixed: belief = w × belief_or + (1-w) × belief_and
//
// ── SYNERGY ────────────────────────────────────────────────────────
//
//   When ≥2 parents fire above SYNERGY_THRESHOLD:
//   synergyExp = 1 + SYNERGY_BOOST × min(activeParents - 1, 3)
//   posProduct_syn = posProduct ^ synergyExp
//
// ── RIGIDITY (Phase 2) ─────────────────────────────────────────────
//
//   Message attenuation: q_eff = q × (1 - rigidity × rigidityWeight)
//   rigidityWeight=0 → identical to Phase 1 behavior (default).
//
//   After convergence:
//   deformationCost[v] = computeDeformationCost(state, |belief - prior|)
//   collapseEvents: nodes where effectiveStrain ≥ breakingPoint
//
// ── CONDUCTIVITY (Phase 2 / Phase 3) ───────────────────────────────
//
//   When enablePressure=true: message along edge is multiplied by
//   computeEdgeConductivity(edge, sourceRigidity, targetRigidity)
//
// ── DAMPING ────────────────────────────────────────────────────────
//
//   belief_t = damping × computed_t + (1 - damping) × belief_{t-1}
//
// ── CONVERGENCE ────────────────────────────────────────────────────
//
//   max |belief_t - belief_{t-1}| < tolerance over all nodes
//
// ====================================================================

// ── Activation scale per edge type ─────────────────────────────────
const TYPE_SCALE_POS: Record<string, number> = {
  determines:   1.00,
  implies:      1.00,
  enables:      0.95,
  exploits:     0.90,
  reveals:      0.82,
  influences:   0.62,
  suggests:     0.50,
  weakens:      0.00,
  // Phase 2-5 edge types
  resists:      0.30,
  pressures:    0.70,
  constrains:   0.80,
  temporalNext: 0.50,
}

const SCALE_NEG_WEAKENS   = 0.78
const DEFAULT_PRIOR       = 0.20
const DEFAULT_SENSITIVITY = 0.85
const DEFAULT_SIGMA       = 0.30
const SYNERGY_THRESHOLD   = 0.45
const SYNERGY_BOOST       = 0.28
const EPSILON             = 1e-9

function clamp01(x: number): number {
  return x < EPSILON ? 0 : x > 1 - EPSILON ? 1 : x
}

function binaryEntropy(p: number): number {
  if (p <= 0 || p >= 1) return 0
  return -(p * Math.log2(p) + (1 - p) * Math.log2(1 - p))
}

function computeGraphEntropy(beliefs: Map<string, number>): number {
  let H = 0
  for (const [, b] of beliefs) H += binaryEntropy(b)
  return H
}

// ── Configuration ──────────────────────────────────────────────────

export interface BPConfig {
  /** Nombre maximum d'itérations (défaut: 40) */
  maxIter: number
  /** Facteur de damping anti-oscillation [0,1) (défaut: 0.80) */
  damping: number
  /** Seuil de convergence max|Δbelief| (défaut: 0.0006) */
  tolerance: number
  /** Mode de gate global (override per-node gateType si spécifié) */
  gateMode?: 'noisyOR' | 'noisyAND' | 'synergy'
  /** Poids de la rigidité dans le message passing [0,1]
   *  0.0 = Phase 1 behavior (rigidity inactive)
   *  >0  = Phase 2 rigidity attenuation active */
  rigidityWeight: number
  /** États de rigidité par nœud (Phase 2) — fourni par Graph.rigidityStates */
  rigidityStates?: Map<string, RigidityState>
  /** Activer la conductivité des arêtes dans BP (Phase 3, défaut: false) */
  enablePressure: boolean
  /** Decay de pression par itération (Phase 3, défaut: 0.9) */
  pressureDecay: number
  /** Nœuds observés sur le canvas — initialisés à leur prior, pas à 0 */
  observedNodeIds?: Set<string>
  /** Ordonnancement des messages (défaut: sequential) */
  scheduling: 'sequential' | 'residual'
  /** Callback après chaque itération — pour instrumentation Phase 5 */
  onIterationComplete?: (iter: number, beliefs: Map<string, number>, maxDelta: number) => void

  // ── Phase 6 — Hybrid BP/TDA ─────────────────────────────────────────────
  /** Matrice d'affinité sémantique (SemanticAffinity). Si absente : BP standard. */
  affinityMatrix?: AffinityMatrix
  /** Poids du canal affinité [0, 0.25] — contrôle l'influence du contexte sémantique.
   *  0 = BP standard. 0.12 = hybride équilibré (défaut). 0.25 = contexte dominant. */
  affinityScale?: number
  /** Poids d'affinité minimal pour activer le canal (défaut: 0.20) */
  affinityMinWeight?: number
  /** Résultat topologique du run BP précédent (T-1). Modifie rigidityAttenuation
   *  par nœud : les nœuds core résistent plus au pull contextuel. */
  topologyResult?: TopologyResult
  /** Amplitude du boost adaptatif de affinityScale quand BP oscille [0, maxBoostCap].
   *  0 = désactivé (défaut). 0.5 = boost modéré sur oscillation. */
  topologyBoost?: number
  /** Cap absolu sur le boost dynamique (défaut: 0.5) */
  maxBoostCap?: number
}

export const DEFAULT_BP_CONFIG: BPConfig = {
  maxIter:          40,
  damping:          0.80,
  tolerance:        0.0006,
  rigidityWeight:   0.0,
  enablePressure:   false,
  pressureDecay:    0.9,
  scheduling:       'sequential',
  affinityScale:    0.12,
  affinityMinWeight: 0.20,
}

// ── Résultat ───────────────────────────────────────────────────────

export interface BPResult {
  beliefs:          Map<string, number>   // nodeId → posterior [0,1]
  iterations:       number
  converged:        boolean
  residuals:        Map<string, number>   // nodeId → résidu final
  entropy:          number                // entropie globale H(beliefs)
  maxBeliefDelta:   number                // plus grand Δbelief au dernier iter

  // Phase 2 — Résistance
  deformationCosts: Map<string, number>   // nodeId → coût de déformation (0 si rigidityWeight=0)
  collapseEvents:   string[]              // nodeIds ayant dépassé leur breakingPoint
}

/** Alias de compatibilité — BeliefMap = BPResult */
export type BeliefMap = BPResult

// ── Fonction principale ────────────────────────────────────────────

export function runBP(
  graph: Graph,
  evidence: Evidence[],
  config: Partial<BPConfig> = {}
): BPResult {
  const cfg: BPConfig = { ...DEFAULT_BP_CONFIG, ...config }
  const nodes = graph.getAllNodes()

  const emptyResult: BPResult = {
    beliefs: new Map(), iterations: 0, converged: true,
    residuals: new Map(), entropy: 0, maxBeliefDelta: 0,
    deformationCosts: new Map(), collapseEvents: [],
  }

  if (nodes.length === 0) return emptyResult

  const observedNodeIds = cfg.observedNodeIds
  if (observedNodeIds && observedNodeIds.size === 0 && evidence.length === 0) return emptyResult

  // ── Evidence partitioning ─────────────────────────────────────────
  const posEvidenceMap = new Map<string, number>()
  const negEvidenceMap = new Map<string, number>()

  for (const e of evidence) {
    if (e.value === false) {
      negEvidenceMap.set(e.nodeId, clamp01(e.confidence))
    } else {
      posEvidenceMap.set(e.nodeId, clamp01(e.confidence))
    }
  }

  // ── Pre-compute per-node parameters ──────────────────────────────
  const truePriors = new Map<string, number>()
  const leaks      = new Map<string, number>()
  const sensMap    = new Map<string, number>()
  const rigMap     = new Map<string, number>()

  for (const node of nodes) {
    const p    = clamp01(node.prior       ?? DEFAULT_PRIOR)
    const sig  = clamp01(node.sigma       ?? DEFAULT_SIGMA)
    const sens = clamp01(node.sensitivity ?? DEFAULT_SENSITIVITY)
    // Rigidity: from rigidityStates if available, else from node field, else 0
    const rigState = cfg.rigidityStates?.get(node.id)
    const rig  = clamp01(rigState?.rigidity ?? node.rigidity ?? 0)

    truePriors.set(node.id, p)
    sensMap.set(node.id, sens)
    rigMap.set(node.id, rig)
    leaks.set(node.id, clamp01(p + sig * 0.4 * (0.5 - p)))
  }

  // ── Initialize beliefs ────────────────────────────────────────────
  const beliefs = new Map<string, number>()
  for (const node of nodes) {
    const isObserved = !observedNodeIds || observedNodeIds.has(node.id)
    beliefs.set(node.id, isObserved ? truePriors.get(node.id)! : 0)
  }
  for (const [nodeId, conf] of posEvidenceMap) beliefs.set(nodeId, conf)
  for (const [nodeId, conf] of negEvidenceMap) {
    const prior = truePriors.get(nodeId) ?? DEFAULT_PRIOR
    beliefs.set(nodeId, clamp01(prior * (1 - conf)))
  }

  let iterations    = 0
  let converged      = false
  let maxBeliefDelta = 0
  let prevMaxDelta   = 0    // Phase 6 : convergence tracking pour α adaptatif
  let smoothedRatio  = 0.0  // EMA du ratio — démarre à 0 (pas de boost avant données de convergence)

  // ── Main iteration loop ───────────────────────────────────────────
  for (let iter = 0; iter < cfg.maxIter; iter++) {
    iterations++
    const prev = new Map(beliefs)

    // ── Phase 6 : ratio de convergence → dynamic affinityScale ───────
    // rawRatio ≈ 1 : BP stagne/oscille → topology boost actif
    // rawRatio → 0 : BP converge vite   → topology observe
    // EMA (α=0.30) pour éviter que boost oscille en phase avec BP
    const iterPrevDelta = maxBeliefDelta
    const rawRatio = iter > 0 && prevMaxDelta > 1e-6
      ? Math.min(1, iterPrevDelta / prevMaxDelta)
      : 1.0
    smoothedRatio = 0.70 * smoothedRatio + 0.30 * rawRatio
    prevMaxDelta  = iterPrevDelta

    const dynamicAffinityScale = (cfg.topologyBoost && cfg.topologyBoost > 0)
      ? (cfg.affinityScale ?? 0) * (1 + Math.min(
          smoothedRatio * cfg.topologyBoost,
          cfg.maxBoostCap ?? 0.5,
        ))
      : (cfg.affinityScale ?? 0)

    maxBeliefDelta = 0

    // Update order
    let updateOrder: typeof nodes
    if (cfg.scheduling === 'residual' && iter > 0) {
      updateOrder = [...nodes].sort((a, b) => {
        const ra = Math.abs((beliefs.get(a.id) ?? 0) - (prev.get(a.id) ?? 0))
        const rb = Math.abs((beliefs.get(b.id) ?? 0) - (prev.get(b.id) ?? 0))
        return rb - ra
      })
    } else {
      updateOrder = nodes
    }

    for (const node of updateOrder) {
      if (posEvidenceMap.has(node.id) || negEvidenceMap.has(node.id)) continue

      const isObserved  = !observedNodeIds || observedNodeIds.has(node.id)
      const prior       = isObserved ? truePriors.get(node.id)! : 0
      const leak        = isObserved ? leaks.get(node.id)!      : 0
      const sensitivity = sensMap.get(node.id)!
      const rigidity    = rigMap.get(node.id)!
      const gateType    = node.gateType  ?? 'or'
      const gateWeight  = node.gateWeight ?? 0.5
      const incoming    = graph.getIncomingEdges(node.id)

      if (incoming.length === 0) {
        beliefs.set(node.id, prior)
        continue
      }

      const posEdges = incoming.filter(e => e.type !== 'weakens')
      const negEdges = incoming.filter(e => e.type === 'weakens')

      // ── Rigidity attenuation (Phase 2 + Phase 6 composite) ────────
      // Phase 6 : si topologyResult présent, les nœuds core (haute
      // persistence + cycle) acquièrent une rigidité structurelle
      // indépendante du rigidityWeight — ils résistent au pull affinité.
      let rigidityAttenuation: number
      if (cfg.topologyResult) {
        const topoProfile = cfg.topologyResult.profiles.get(node.id)
        const topoRigidity = topoProfile
          ? clamp01(topoProfile.persistence * (topoProfile.inCycle ? 1.4 : 1.0))
          : 0
        // Max entre rigidité manuelle (gated by weight) et rigidité topologique
        const structuralRigidity = clamp01(Math.max(rigidity * cfg.rigidityWeight, topoRigidity))
        rigidityAttenuation = 1 - structuralRigidity
      } else {
        // Comportement Phase 1/2 préservé intégralement
        rigidityAttenuation = 1 - rigidity * cfg.rigidityWeight
      }

      // ── Synergy ───────────────────────────────────────────────────
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

        // Phase 3: conductivity scaling when enablePressure=true
        let conductivityScale = 1.0
        if (cfg.enablePressure && cfg.rigidityStates) {
          const srcRig = cfg.rigidityStates.get(edge.source)
          const tgtRig = cfg.rigidityStates.get(node.id)
          if (srcRig && tgtRig) {
            conductivityScale = computeEdgeConductivity(edge, srcRig, tgtRig)
          }
        }

        const typeScale = TYPE_SCALE_POS[edge.type] ?? 0.50
        const q  = typeScale * rigidityAttenuation * conductivityScale
        const qi = clamp01(q * edge.weight * srcBelief)
        posProduct *= (1 - qi)
      }
      const posProductSyn = posEdges.length > 0 ? Math.pow(posProduct, synergyExponent) : 1.0
      const beliefOr = clamp01(1 - (1 - leak) * posProductSyn)

      // ── AND gate ──────────────────────────────────────────────────
      let beliefAnd = leak
      if (posEdges.length > 0) {
        let logSum = 0
        for (const edge of posEdges) {
          const srcBelief = prev.get(edge.source) ?? 0
          let conductivityScaleAnd = 1.0
          if (cfg.enablePressure && cfg.rigidityStates) {
            const srcRig = cfg.rigidityStates.get(edge.source)
            const tgtRig = cfg.rigidityStates.get(node.id)
            if (srcRig && tgtRig) {
              conductivityScaleAnd = computeEdgeConductivity(edge, srcRig, tgtRig)
            }
          }
          const q = (TYPE_SCALE_POS[edge.type] ?? 0.50) * rigidityAttenuation * conductivityScaleAnd
          const activation = q * edge.weight * srcBelief
          logSum += Math.log(Math.max(EPSILON, activation))
        }
        const andMean = Math.exp(logSum / posEdges.length)
        beliefAnd = clamp01(leak + (1 - leak) * andMean)
      }

      // ── Combine gates ─────────────────────────────────────────────
      const effectiveGate = cfg.gateMode === 'noisyAND' ? 'and'
        : cfg.gateMode === 'noisyOR' ? 'or'
        : gateType
      let activated: number
      if (effectiveGate === 'and') {
        activated = beliefAnd
      } else if (effectiveGate === 'mixed') {
        activated = clamp01(gateWeight * beliefOr + (1 - gateWeight) * beliefAnd)
      } else {
        activated = beliefOr
      }

      // ── Inhibition ────────────────────────────────────────────────
      let negProduct = 1.0
      for (const edge of negEdges) {
        const srcBelief = prev.get(edge.source) ?? 0
        const inh = clamp01(SCALE_NEG_WEAKENS * edge.weight * srcBelief)
        negProduct *= (1 - inh)
      }

      let rawBelief = clamp01(activated * negProduct)

      // ── Phase 6 : Canal sémantique (Hybrid BP/TDA) ────────────────
      if (cfg.affinityMatrix && dynamicAffinityScale > 0) {
        const affinRow = cfg.affinityMatrix.get(node.id)
        const minW = cfg.affinityMinWeight ?? 0.20
        if (affinRow && affinRow.size > 0) {
          let contextSum = 0, totalAffinity = 0
          for (const [neighborId, w] of affinRow) {
            if (w < minW) continue
            const nb = prev.get(neighborId)   // Jacobi : snapshot début d'itération, cohérent avec KB edges
            if (nb === undefined) continue
            contextSum += w * nb; totalAffinity += w
          }
          if (totalAffinity > 0) {
            const contextBelief = contextSum / totalAffinity
            const α = clamp01(dynamicAffinityScale * rigidityAttenuation)
            rawBelief = clamp01((1 - α) * rawBelief + α * contextBelief)
          }
        }
      }

      // ── Sensitivity modulation ────────────────────────────────────
      const computed = clamp01(prior + sensitivity * (rawBelief - prior))

      // ── Damping ───────────────────────────────────────────────────
      const prevBelief = prev.get(node.id) ?? 0
      const newBelief  = clamp01(cfg.damping * computed + (1 - cfg.damping) * prevBelief)
      beliefs.set(node.id, newBelief)
    }

    // ── Re-clamp evidence ─────────────────────────────────────────
    for (const [nodeId, conf] of posEvidenceMap) beliefs.set(nodeId, conf)
    for (const [nodeId, conf] of negEvidenceMap) {
      const prior = truePriors.get(nodeId) ?? DEFAULT_PRIOR
      beliefs.set(nodeId, clamp01(prior * (1 - conf)))
    }

    // ── Convergence check ─────────────────────────────────────────
    for (const [id, val] of beliefs) {
      const delta = Math.abs(val - (prev.get(id) ?? 0))
      if (delta > maxBeliefDelta) maxBeliefDelta = delta
    }

    cfg.onIterationComplete?.(iter, new Map(beliefs), maxBeliefDelta)

    if (maxBeliefDelta < cfg.tolerance) {
      converged = true
      break
    }
  }

  // ── Post-convergence: residuals ───────────────────────────────────
  const residuals = new Map<string, number>()
  for (const node of nodes) {
    const b = beliefs.get(node.id) ?? 0
    residuals.set(node.id, Math.abs(b - (truePriors.get(node.id) ?? DEFAULT_PRIOR)))
  }

  const entropy = computeGraphEntropy(beliefs)

  // ── Post-convergence: Phase 2 deformation analysis ────────────────
  const deformationCosts = new Map<string, number>()
  const collapseEvents:   string[] = []

  if (cfg.rigidityWeight > 0 || cfg.rigidityStates) {
    for (const node of nodes) {
      const rigState = cfg.rigidityStates?.get(node.id)
      if (!rigState) continue

      const beliefFinal = beliefs.get(node.id) ?? 0
      const prior       = truePriors.get(node.id) ?? DEFAULT_PRIOR
      const beliefDelta = Math.abs(beliefFinal - prior)

      // Deformation cost: how much rigidity "resisted" the observed delta
      const cost = computeDeformationCost(rigState, beliefDelta)
      deformationCosts.set(node.id, cost)

      // Collapse detection: strain that penetrated the rigidity
      if (!rigState.collapsed) {
        const effectiveStrain = beliefDelta * (1 - rigState.rigidity)
        if (effectiveStrain >= rigState.breakingPoint) {
          collapseEvents.push(node.id)
        }
      }
    }
  }

  return { beliefs, iterations, converged, residuals, entropy, maxBeliefDelta, deformationCosts, collapseEvents }
}

// ── Class wrapper — garde la compatibilité avec new BeliefPropagation().propagate() ──

export class BeliefPropagation {
  constructor(
    private readonly maxIter:   number = 40,
    private readonly threshold: number = 0.0006
  ) {}

  propagate(graph: Graph, evidence: Evidence[], observedNodeIds?: Set<string>): BPResult {
    return runBP(graph, evidence, {
      maxIter:          this.maxIter,
      tolerance:        this.threshold,
      observedNodeIds,
    })
  }
}

// ── Utilities ─────────────────────────────────────────────────────

export function logit(p: number): number {
  const c = Math.max(EPSILON, Math.min(1 - EPSILON, p))
  return Math.log(c / (1 - c))
}

export function sigmoid(lo: number): number {
  if (lo >= 0) { const e = Math.exp(-lo); return 1 / (1 + e) }
  const e = Math.exp(lo); return e / (1 + e)
}

export function jeffreysInterval(k: number, n: number): [number, number] {
  if (n === 0) return [0, 1]
  const p  = (k + 0.5) / (n + 1)
  const se = Math.sqrt(p * (1 - p) / (n + 1))
  return [Math.max(0, p - 1.96 * se), Math.min(1, p + 1.96 * se)]
}
