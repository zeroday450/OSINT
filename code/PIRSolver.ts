// ============================================================
// EIDOLON PIR Solver
// 
//
// RÔLE
// ════
// Résoudre : argmax_{pression} ΔH(beliefs | pression)
//
// Pour chaque candidat (noeud × direction), le solveur exécute
// un simulatePressure() (dry-run, zéro écriture graphe) et
// calcule le gain d'information attendu :
//
//   gain = H(beliefs_avant) - H(beliefs_après_prédit)
//
// Le candidat avec le gain maximal est la prochaine pression
// optimale à appliquer selon le critère de réduction d'entropie.
//
// SIGNAL DE CONTRAINTE
// ════════════════════
// Un noeud qui résiste (|delta| faible sous forte pression) est
// une contrainte incompressible. Le solveur calcule un score
// de résistance par noeud sur l'ensemble des simulations.
//
//   constraintSignal[v] = 1 - (totalDelta[v] / expectedMax)
//
// constraintSignal proche de 1 → contrainte structurelle.
// constraintSignal proche de 0 → dépendance superficielle.
//
// INVARIANTS
// ══════════
// - Aucune écriture sur le graphe réel (simulatePressure only).
// - Entropie calculée uniquement sur les noeuds canvas (actifs).
// - Le solveur est déterministe et pur (pas de LLM).
// ============================================================

import type { PressureEngine } from './PressureEngine'
import { PressureDirection, PressureProfile, PressureSource } from '../types/pressure'
import type { BeliefMap } from './BeliefPropagation'
import { createPressure } from './PressureGenerator'
import type { TopologyResult } from './BeliefTopology'
import { topologicalDisruptionScore } from './BeliefTopology'
import type { AffinityMatrix } from './SemanticAffinity'

export interface PIRCandidate {
  nodeId:           string
  nodeLabel:        string
  nodeCategory:     string
  direction:        PressureDirection
  magnitude:        number
  profile:          PressureProfile

  // Métriques d'information
  entropyGain:      number    // H(avant) - H(après prédit) — gain en certitude
  totalDelta:       number    // Σ|delta_canvas| — énergie totale de réponse
  constraintSignal: number    // résistance du nœud cible [0=volatile, 1=contrainte]
  predictedDeltas:  Map<string, number>
  confidence:       number

  // Phase 6 — Score topologique hybride
  topologicalScore: number    // disruption topologique prédite [0,1]
  combinedScore:    number    // 0.70·entropyGain_norm + 0.30·topologicalScore_norm
}

export interface PIRSolverResult {
  timestamp:           number
  currentEntropy:      number    // H(beliefs) au moment du solve
  candidates:          PIRCandidate[]   // triés par entropyGain desc
  topCandidate:        PIRCandidate | null

  // Profils de résistance par nœud (sur toutes simulations)
  resistanceMap:       Map<string, number>  // nodeId → constraintSignal moyen
  incompressibleNodes: string[]   // constraintSignal > CONSTRAINT_THRESHOLD
  volatileNodes:       string[]   // constraintSignal < VOLATILE_THRESHOLD
}

export interface PIRSolverOptions {
  magnitudes?:        number[]           // magnitudes à tester (défaut: [0.3, 0.6, 0.9])
  directions?:        PressureDirection[] // directions à tester (défaut: toutes)
  profile?:           PressureProfile    // profil temporel à utiliser
  canvasNodeIds?:     Set<string>        // restreindre aux nœuds canvas (entropie + résistance)
  maxCandidates?:     number             // nombre max de candidats retournés (défaut: 20)
  topology?:          TopologyResult     // Phase 6 — TDA pour scoring hybride
  affinityMatrix?:    AffinityMatrix     // Phase 6 — affinités sémantiques
}

const CONSTRAINT_THRESHOLD = 0.65  // signal > seuil → incompressible
const VOLATILE_THRESHOLD   = 0.25  // signal < seuil → volatile
const DEFAULT_MAGNITUDES   = [0.3, 0.6, 0.9]
const DEFAULT_DIRECTIONS   = [
  PressureDirection.confirm,
  PressureDirection.contradict,
  PressureDirection.ambiguate,
  PressureDirection.polarize,
]

// ── Entropie de Shannon sur un ensemble de beliefs ───────────────────────────
// H = -Σ p_i * log2(p_i)  (0·log2(0) = 0 par convention)
function shannonEntropy(beliefs: Iterable<number>): number {
  let h = 0
  for (const p of beliefs) {
    const q = Math.max(1e-10, Math.min(1 - 1e-10, p))
    h -= q * Math.log2(q) + (1 - q) * Math.log2(1 - q)
  }
  return h
}

// ── Entropie restreinte aux nœuds canvas ─────────────────────────────────────
function canvasEntropy(beliefMap: BeliefMap, canvasIds: Set<string>): number {
  const values: number[] = []
  for (const nodeId of canvasIds) {
    const b = beliefMap.beliefs.get(nodeId)
    if (b !== undefined) values.push(b)
  }
  return shannonEntropy(values)
}

// ── Entropie prédite après application d'un delta ────────────────────────────
function predictedEntropy(
  beliefMap: BeliefMap,
  canvasIds: Set<string>,
  deltas: Map<string, number>,
): number {
  const values: number[] = []
  for (const nodeId of canvasIds) {
    const b = beliefMap.beliefs.get(nodeId) ?? 0
    const d = deltas.get(nodeId) ?? 0
    values.push(Math.max(0, Math.min(1, b + d)))
  }
  return shannonEntropy(values)
}

// ── Solveur PIR ───────────────────────────────────────────────────────────────

export class PIRSolver {
  constructor(private readonly engine: PressureEngine) {}

  solve(
    targetCandidates: Array<{ id: string; label: string; category: string }>,
    beliefMap:        BeliefMap,
    evidence:         import('../types').Evidence[],
    opts:             PIRSolverOptions = {},
  ): PIRSolverResult {
    const magnitudes  = opts.magnitudes   ?? DEFAULT_MAGNITUDES
    const directions  = opts.directions   ?? DEFAULT_DIRECTIONS
    const profile     = opts.profile      ?? PressureProfile.impulse
    const canvasIds   = opts.canvasNodeIds ?? new Set(targetCandidates.map(n => n.id))
    const maxCandidates = opts.maxCandidates ?? 20
    const topology    = opts.topology
    const affinityMatrix = opts.affinityMatrix

    const H0 = canvasEntropy(beliefMap, canvasIds)

    // Accumulate resistance signals per node across all simulations
    const nodeResistanceSamples = new Map<string, number[]>()

    const allCandidates: PIRCandidate[] = []

    for (const node of targetCandidates) {
      if (!nodeResistanceSamples.has(node.id)) {
        nodeResistanceSamples.set(node.id, [])
      }

      // ── Phase 6 : Score topologique — calculé une seule fois par nœud ────────
      const topoScore = topology && affinityMatrix
        ? topologicalDisruptionScore(node.id, topology, affinityMatrix)
        : 0

      for (const direction of directions) {
        for (const magnitude of magnitudes) {
          const event = createPressure({
            targetNodeIds: [node.id],
            direction,
            magnitude,
            profile,
            source: PressureSource.solver,
            metadata: { pirSolve: true },
          })

          const sim = this.engine.simulatePressure(event, evidence)

          // Entropy gain from predicted deltas
          const Hafter = predictedEntropy(beliefMap, canvasIds, sim.predictedDeltas)
          const entropyGain = H0 - Hafter

          // Total absolute delta across canvas nodes
          let totalDelta = 0
          for (const nodeId of canvasIds) {
            const d = sim.predictedDeltas.get(nodeId)
            if (d !== undefined) totalDelta += Math.abs(d)
          }

          // Constraint signal for this node at this magnitude
          // High resistance = small delta relative to magnitude applied
          const targetDelta = Math.abs(sim.predictedDeltas.get(node.id) ?? 0)
          // Expected maximum delta if node had zero resistance
          const expectedMax = magnitude
          const constraintSignal = 1 - Math.min(1, targetDelta / Math.max(expectedMax, 0.01))

          nodeResistanceSamples.get(node.id)!.push(constraintSignal)

          allCandidates.push({
            nodeId:           node.id,
            nodeLabel:        node.label,
            nodeCategory:     node.category,
            direction,
            magnitude,
            profile,
            entropyGain,
            totalDelta,
            constraintSignal,
            predictedDeltas:  sim.predictedDeltas,
            confidence:       sim.confidence,
            topologicalScore: topoScore,
            combinedScore:    0,  // computed after normalization below
          })
        }
      }
    }

    // ── Phase 6 : Normaliser et combiner les scores ───────────────────────────
    // Normalisation min-max sur [0,1] : (val - min) / (max - min)
    // Évite les eNorm extrêmement négatifs quand tous les gains sont < 0.
    const minEntropy = allCandidates.reduce((m, c) => Math.min(m, c.entropyGain), Infinity)
    const maxEntropy = allCandidates.reduce((m, c) => Math.max(m, c.entropyGain), -Infinity)
    const minTopo    = allCandidates.reduce((m, c) => Math.min(m, c.topologicalScore), Infinity)
    const maxTopo    = allCandidates.reduce((m, c) => Math.max(m, c.topologicalScore), -Infinity)
    const eRange     = maxEntropy - minEntropy
    const tRange     = maxTopo    - minTopo

    for (const c of allCandidates) {
      const eNorm = eRange > 1e-10 ? (c.entropyGain      - minEntropy) / eRange : 0.5
      const tNorm = tRange > 1e-10 ? (c.topologicalScore - minTopo)    / tRange : 0.5
      c.combinedScore = 0.70 * eNorm + 0.30 * tNorm
    }

    // Sort by combinedScore (desc), then entropyGain as tiebreaker
    allCandidates.sort((a, b) =>
      b.combinedScore - a.combinedScore || b.entropyGain - a.entropyGain,
    )

    const candidates = allCandidates.slice(0, maxCandidates)
    const topCandidate = candidates[0] ?? null

    // Compute mean resistance signal per node
    const resistanceMap = new Map<string, number>()
    for (const [nodeId, samples] of nodeResistanceSamples.entries()) {
      if (samples.length === 0) continue
      resistanceMap.set(nodeId, samples.reduce((s, v) => s + v, 0) / samples.length)
    }

    const incompressibleNodes: string[] = []
    const volatileNodes:       string[] = []

    for (const node of targetCandidates) {
      const sig = resistanceMap.get(node.id) ?? 0
      if (sig > CONSTRAINT_THRESHOLD)  incompressibleNodes.push(node.id)
      if (sig < VOLATILE_THRESHOLD)    volatileNodes.push(node.id)
    }

    return {
      timestamp:     Date.now(),
      currentEntropy: H0,
      candidates,
      topCandidate,
      resistanceMap,
      incompressibleNodes,
      volatileNodes,
    }
  }
}
