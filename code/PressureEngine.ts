// ============================================================
// EIDOLON Pressure Engine
// 
//
// RÔLE
// ════
// Orchestrer l'application de PressureEvents sur le graphe :
//   1. Calculer le prior shift sur les nœuds cibles (filtré par rigidité)
//   2. Propager la vague de pression par BFS via conductivité des arêtes
//   3. Mettre à jour les RigidityStates (strain, collapse)
//   4. Exécuter runBP pour recalculer les beliefs globaux
//   5. Retourner un PressureResult avec le delta complet
//
// INVARIANTS
// ══════════
//   - Le moteur est pur du point de vue analytique.
//     Le LLM ne touche pas à ce module.
//   - prior ⊥ rigidity : deux axes orthogonaux.
//     La pression modifie le prior de l'entité (notre estimation),
//     la rigidité filtre combien de cette pression passe réellement.
//   - reset() restaure les priors originaux (snapshot à l'init).
//   - simulatePressure() n'écrit jamais sur le graphe réel.
//
// MÉCANIQUE DE PRIOR SHIFT
// ════════════════════════
//   confirm    : shift = +magnitude × (1 - prior)   → push toward 1
//   contradict : shift = -magnitude × prior          → push toward 0
//   ambiguate  : shift = magnitude × (0.5 - prior)  → push toward 0.5
//   polarize   : if prior > 0.5 → like confirm
//                if prior < 0.5 → like contradict
//                (amplifier la croyance dominante)
//
// PROPAGATION DE LA VAGUE
// ═══════════════════════
//   BFS depuis les targetNodes.
//   Pression à chaque hop : p_next = p_current × conductivity(edge) × pressureDecay
//   Stop quand p < tolerance.
//   Chaque nœud visité reçoit son propre prior shift (filtré par sa rigidité).
//
// ============================================================

import { runBP, type BPConfig, DEFAULT_BP_CONFIG } from './BeliefPropagation'
import {
  initializeRigidity,
  applyStrain,
  computeEffectivePriorShift,
  computeEdgeConductivity,
  type RigidityState,
} from './RigidityModel'
import { expandProfile, effectiveSteps } from './PressureProfiles'
import { Graph } from '../graph/Graph'
import type { Evidence } from '../types'
import type {
  PressureEvent,
  PressureResult,
  PressureSimulation,
  WavePropagationLog,
  WaveHop,
} from '../types/pressure'
import { PressureDirection } from '../types/pressure'

// ── Tolérance de propagation (pression < ce seuil → stop BFS) ────────
const PROPAGATION_TOLERANCE = 0.005

// ── EdgeType résistant — barrière structurelle dans la vague BFS ──────
// Un lien 'resists' compose deux fois la rigidité du nœud cible :
//   conductivity_effective = computeEdgeConductivity(e, src, tgt)
//                          × (1 - targetRigidity)
// Pour un nœud idéologique (rigidity=0.87) :
//   facteur normal ≈ 0.13  → facteur résistance ≈ 0.13 × 0.13 = 0.017
// La pression traversant une barrière idéologique est quasi-nulle.
// Pour une vulnérabilité (rigidity=0.25) :
//   facteur normal ≈ 0.75  → facteur résistance ≈ 0.75 × 0.75 = 0.56
// La pression traverse librement — une vulnérabilité ne bloque pas.
const RESISTS_BARRIER_FACTOR = (targetRigidity: number): number =>
  1 - targetRigidity

// ── Conductivité BFS par type d'arête ────────────────────────────────
//
// Distinct de computeEdgeConductivity (BP) qui utilise edge.conductivity ?? 0.5
// — aucune arête KB n'ayant ce champ, toutes étaient divisées par 2 à chaque hop,
// limitant la propagation à 3-4 hops maximum.
//
// Ici, la conductivité encode la sémantique du type d'arête :
//   determines/implies  → quasi-transparent (0.92)
//   enables/exploits    → fort (0.82)
//   reveals/influences  → modéré (0.70)
//   suggests            → faible (0.50)
//   weakens             → propagation inversée (0.72) — direction déjà inversée
//   resists             → très faible (0.35) + RESISTS_BARRIER_FACTOR
//
// Avec weight=0.8, avgRig=0.25 sur un lien 'implies' :
//   conductivity = 0.8 × 0.92 × 0.75 = 0.552
//   nextPressure = pressure × 0.552     (au lieu de × 0.27 avant)
// → 8 hops au lieu de 4.
//
const BFS_TYPE_DECAY: Record<string, number> = {
  determines:   0.92,
  implies:      0.92,
  enables:      0.82,
  exploits:     0.80,
  reveals:      0.70,
  influences:   0.65,
  suggests:     0.50,
  weakens:      0.72,   // direction déjà inversée par invertDirectionForEdge
  resists:      0.35,   // + RESISTS_BARRIER_FACTOR ci-dessous
  pressures:    0.88,
  constrains:   0.68,
  temporalNext: 0.55,
}

function bfsEdgeConductivity(
  edge:    { type: string; weight?: number },
  srcRig:  RigidityState,
  tgtRig:  RigidityState,
): number {
  const typeDecay = BFS_TYPE_DECAY[edge.type] ?? 0.60
  const avgRig    = (srcRig.rigidity + tgtRig.rigidity) / 2
  return (edge.weight ?? 1.0) * typeDecay * (1 - avgRig)
}

// ── Inversion de direction pour les arêtes sémantiquement négatives ───
//
// Une arête 'weakens' (ou 'resists') crée une relation causale inverse :
//   A --weakens--> B  signifie  si A↑ alors B↓
//
// Le BFS doit donc inverser la direction de pression quand il traverse
// une telle arête, sinon il confirme B alors qu'il devrait le contredire,
// ce qui crée un conflit que BP doit ensuite corriger — produisant des
// deltas négatifs inattendus sur tous les nœuds reliés par des liens forts.
//
// AMBIGUATE : inchangé — pousser vers 0.5 reste cohérent quelle que soit
//             la polarité de l'arête (l'incertitude ne s'inverse pas).
// POLARIZE  : inchangé — le polarize amplifie la conviction dominante du
//             nœud cible ; c'est celle-ci, pas la direction de l'arête,
//             qui détermine vers où pousser.
//
function invertDirectionForEdge(
  dir:      PressureDirection,
  edgeType: string,
): PressureDirection {
  if (edgeType === 'weakens' || edgeType === 'resists') {
    if (dir === PressureDirection.confirm)    return PressureDirection.contradict
    if (dir === PressureDirection.contradict) return PressureDirection.confirm
  }
  return dir
}

// ── Calcul du prior shift brut selon la direction ────────────────────

function computePriorShift(
  currentPrior: number,
  magnitude:    number,
  direction:    PressureDirection,
): number {
  switch (direction) {
    case PressureDirection.confirm:
      // push toward 1 : plus le prior est déjà haut, moins ça change
      return magnitude * (1 - currentPrior)

    case PressureDirection.contradict:
      // push toward 0 : plus le prior est bas, moins ça change
      return -magnitude * currentPrior

    case PressureDirection.ambiguate:
      // push toward 0.5 (incertitude maximale)
      return magnitude * (0.5 - currentPrior)

    case PressureDirection.polarize:
      // amplifie la conviction dominante
      if (currentPrior > 0.5) {
        return magnitude * (1 - currentPrior)   // pousse vers 1
      } else if (currentPrior < 0.5) {
        return -magnitude * currentPrior         // pousse vers 0
      }
      return 0  // prior = 0.5 exactement → équilibre instable, pas de shift

    default:
      return 0
  }
}

function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x
}

// ── BFS wave propagation ──────────────────────────────────────────────

interface WaveEntry {
  nodeId:       string
  pressure:     number            // magnitude résiduelle à ce nœud
  hopIndex:     number
  fromNodeId:   string
  edgeId:       string
  conductivity: number
  direction:    PressureDirection  // direction locale — peut être inversée par une arête weakens
}

function propagateWave(
  graph:       Graph,
  seedNodeIds: string[],
  seedPressure: number,
  pressureDecay: number,
  direction:   PressureDirection,
  priorOverrides: Map<string, number>,  // sortie : priors modifiés
  rigiditySnapshots: Map<string, RigidityState>,  // sortie : états rigidité mis à jour
  hops: WaveHop[],
): void {
  const visited = new Set<string>()
  const queue: WaveEntry[] = []

  for (const nodeId of seedNodeIds) {
    const node = graph.getNode(nodeId)
    if (!node) continue
    queue.push({
      nodeId,
      pressure:    seedPressure,
      hopIndex:    0,
      fromNodeId:  'source',
      edgeId:      '',
      conductivity: 1.0,
      direction,   // direction d'origine pour les nœuds seeds
    })
  }

  while (queue.length > 0) {
    const entry = queue.shift()!
    const { nodeId, pressure, hopIndex, fromNodeId, edgeId, conductivity, direction: localDir } = entry

    if (visited.has(nodeId)) continue
    visited.add(nodeId)

    const node = graph.getNode(nodeId)
    if (!node) continue

    // Prior actuel (peut déjà avoir été modifié par un hop précédent dans cette vague)
    const currentPrior = priorOverrides.get(nodeId) ?? node.prior

    // État de rigidité courant
    const rigState = rigiditySnapshots.get(nodeId)
      ?? graph.getRigidity(nodeId)
      ?? initializeRigidity(node)

    // Raw shift → effective shift via rigidité
    // Utilise la direction locale (peut être inversée par une arête weakens/resists traversée)
    const rawShift       = computePriorShift(currentPrior, pressure, localDir)
    const effectiveShift = computeEffectivePriorShift(rigState, rawShift)
    const newPrior       = clamp01(currentPrior + effectiveShift)

    priorOverrides.set(nodeId, newPrior)

    // Mettre à jour le strain de rigidité
    const newRigState = applyStrain(rigState, pressure)
    rigiditySnapshots.set(nodeId, newRigState)

    // Enregistrer le hop
    hops.push({
      fromNodeId,
      toNodeId:    nodeId,
      edgeId,
      depth:       hopIndex,
      conductivity,
      beliefDelta: effectiveShift,  // delta de prior — sera raffiné post-BP
    })

    // Propager aux voisins sortants
    for (const edge of graph.getOutgoingEdges(nodeId)) {
      if (visited.has(edge.target)) continue

      const targetNode = graph.getNode(edge.target)
      if (!targetNode) continue

      const targetRigState = rigiditySnapshots.get(edge.target)
        ?? graph.getRigidity(edge.target)
        ?? initializeRigidity(targetNode)

      // BFS conductivity — sémantique par type, sans le ?? 0.5 arbitraire de BP
      let edgeConductivity = bfsEdgeConductivity(edge, newRigState, targetRigState)
      if (edge.type === 'resists') {
        edgeConductivity *= RESISTS_BARRIER_FACTOR(targetRigState.rigidity)
      }
      const nextPressure = pressure * edgeConductivity

      if (nextPressure < PROPAGATION_TOLERANCE) continue

      // Inverser la direction si l'arête est sémantiquement négative
      const nextDirection = invertDirectionForEdge(localDir, edge.type)

      queue.push({
        nodeId:      edge.target,
        pressure:    nextPressure,
        hopIndex:    hopIndex + 1,
        fromNodeId:  nodeId,
        edgeId:      edge.id,
        conductivity: edgeConductivity,
        direction:   nextDirection,
      })
    }
  }
}

// ── Classe principale ─────────────────────────────────────────────────

export class PressureEngine {
  private readonly graph:        Graph
  private readonly bpConfig:     BPConfig
  private readonly pressureLog:  PressureEvent[]

  // Snapshot des priors originaux — pris à l'init, utilisé par reset()
  private readonly basePriors: Map<string, number>

  constructor(graph: Graph, bpConfig: Partial<BPConfig> = {}) {
    this.graph      = graph
    this.bpConfig   = {
      ...DEFAULT_BP_CONFIG,
      enablePressure:  true,
      rigidityWeight:  1.0,
      ...bpConfig,
    }
    this.pressureLog = []

    // Snapshot des priors à l'initialisation
    this.basePriors = new Map(
      graph.getAllNodes().map(n => [n.id, n.prior]),
    )
  }

  // ── Snapshot du prior d'un nœud (pour sync incrémentale) ───────────
  snapshotPrior(nodeId: string): void {
    const node = this.graph.getNode(nodeId)
    if (node) this.basePriors.set(nodeId, node.prior)
  }

  // ── Application principale d'un PressureEvent ──────────────────────

  applyPressure(event: PressureEvent, evidence: Evidence[] = []): PressureResult {
    const allNodeIds = new Set(this.graph.getAllNodes().map(n => n.id))

    // ── 1. Beliefs avant pression ─────────────────────────────────────
    // IMPORTANT : passer rigidityStates ici pour que bpBefore et bpAfter
    // utilisent le même modèle de conductivité. Sans ça, bpBefore a
    // conductivityScale=1.0 partout et bpAfter a conductivityScale≈0.35
    // (via computeEdgeConductivity post-BFS strain), ce qui produit des
    // deltas négatifs artificiels sur tous les enfants des nœuds visités.
    // À ce point dans le code, this.graph.rigidityStates est l'état PRÉ-BFS.
    const bpBefore = runBP(this.graph, evidence, {
      ...this.bpConfig,
      observedNodeIds: allNodeIds,
      rigidityStates:  this.graph.rigidityStates,   // ← baseline cohérente
    })
    const beliefsBefore = new Map(bpBefore.beliefs)
    const entropyBefore = bpBefore.entropy

    // ── 2. Snapshot rigidité avant (pour calcul de delta) ─────────────
    const rigidityBefore = new Map<string, number>()
    for (const nodeId of allNodeIds) {
      const r = this.graph.getRigidity(nodeId)
      rigidityBefore.set(nodeId, r?.rigidity ?? 0)
    }

    // ── 3. Propagation de la vague (BFS) ──────────────────────────────
    const priorOverrides    = new Map<string, number>()
    const rigiditySnapshots = new Map<string, RigidityState>()
    const hops: WaveHop[]   = []

    const magnitude    = event.pressureVector.magnitude
    const pressureDecay = this.bpConfig.pressureDecay

    // Développer le profil temporel en sous-séquences si non-impulse
    const steps = effectiveSteps(event.pressureVector.profile)
    const magnitudes = expandProfile(event.pressureVector.profile, magnitude, steps)

    for (const stepMagnitude of magnitudes) {
      if (stepMagnitude < PROPAGATION_TOLERANCE) continue
      propagateWave(
        this.graph,
        event.targetNodeIds,
        stepMagnitude,
        pressureDecay,
        event.pressureVector.direction,
        priorOverrides,
        rigiditySnapshots,
        hops,
      )
    }

    // ── 4. Appliquer les mutations sur le graphe réel ─────────────────
    for (const [nodeId, newPrior] of priorOverrides) {
      const node = this.graph.getNode(nodeId)
      if (node) node.prior = newPrior
    }

    for (const [nodeId, newRigState] of rigiditySnapshots) {
      this.graph.setRigidity(nodeId, newRigState)
      if (newRigState.collapsed && !this.graph.isCollapsed(nodeId)) {
        this.graph.markCollapsed(nodeId)
      }
    }

    // ── 5. Re-run BP avec les nouveaux priors + rigidity ─────────────
    const bpAfter = runBP(this.graph, evidence, {
      ...this.bpConfig,
      observedNodeIds: allNodeIds,
      rigidityStates:  this.graph.rigidityStates,
    })
    const beliefsAfter = bpAfter.beliefs
    const entropyAfter = bpAfter.entropy

    // ── 6. Traiter les collapses détectés par BP ──────────────────────
    const allCollapses = new Set<string>()
    for (const nodeId of [...rigiditySnapshots.keys()]) {
      if (this.graph.isCollapsed(nodeId)) allCollapses.add(nodeId)
    }
    for (const nodeId of bpAfter.collapseEvents) {
      if (!this.graph.isCollapsed(nodeId)) this.graph.markCollapsed(nodeId)
      allCollapses.add(nodeId)
    }

    // ── 7. Raffinement des hops : remplacer beliefDelta par delta BP réel ──
    const refinedHops: WaveHop[] = hops.map(hop => ({
      ...hop,
      beliefDelta: (beliefsAfter.get(hop.toNodeId) ?? 0)
                 - (beliefsBefore.get(hop.toNodeId) ?? 0),
    }))

    // ── 8. Calcul des deltas ──────────────────────────────────────────
    const beliefDeltas = new Map<string, number>()
    for (const [id, after] of beliefsAfter) {
      const delta = after - (beliefsBefore.get(id) ?? 0)
      if (Math.abs(delta) > 1e-6) beliefDeltas.set(id, delta)
    }

    const rigidityDeltasObserved = new Map<string, number>()
    for (const [nodeId, rigAfter] of rigiditySnapshots) {
      const rBefore = rigidityBefore.get(nodeId) ?? 0
      const delta   = rigAfter.rigidity - rBefore
      if (Math.abs(delta) > 1e-6) rigidityDeltasObserved.set(nodeId, delta)
    }

    // ── 9. WavePropagationLog ─────────────────────────────────────────
    const visitedNodes = new Set(refinedHops.map(h => h.toNodeId))
    const maxDepth     = refinedHops.reduce((m, h) => Math.max(m, h.depth), 0)
    const totalEnergy  = refinedHops.reduce((s, h) => s + Math.abs(h.beliefDelta), 0)

    const wavePropagation: WavePropagationLog = {
      hops:         refinedHops,
      maxDepth,
      nodesReached: visitedNodes.size,
      totalEnergy,
    }

    // ── 10. Log + retour ──────────────────────────────────────────────
    this.pressureLog.push(event)

    return {
      eventId:                event.id,
      timestamp:              event.timestamp,
      direction:              event.pressureVector.direction,
      beliefsBefore,
      beliefsAfter,
      beliefDeltas,
      entropyBefore,
      entropyAfter,
      collapseEvents:         [...allCollapses],
      rigidityDeltasObserved,
      wavePropagation,
    }
  }

  // ── Simulation à sec (dry run — ne mute pas le graphe) ──────────────

  simulatePressure(event: PressureEvent, evidence: Evidence[] = []): PressureSimulation {
    // Snapshot état courant
    const priorSnapshot    = new Map(this.graph.getAllNodes().map(n => [n.id, n.prior]))
    const rigStateSnapshot = new Map<string, RigidityState>()
    for (const node of this.graph.getAllNodes()) {
      const r = this.graph.getRigidity(node.id)
      if (r) rigStateSnapshot.set(node.id, { ...r })
    }
    // Snapshot exact des nœuds collapsed avant simulation
    const collapsedBefore = new Set(this.graph.getCollapsedNodes())

    // Appliquer sans log — try/finally garantit la restauration même si applyPressure throw
    let result: PressureResult
    try {
      result = this._applyWithoutLog(event, evidence)
    } finally {
      // Restaurer l'état original (s'exécute même en cas d'exception)
      for (const [nodeId, originalPrior] of priorSnapshot) {
        const node = this.graph.getNode(nodeId)
        if (node) node.prior = originalPrior
      }
      for (const [nodeId, originalRig] of rigStateSnapshot) {
        this.graph.setRigidity(nodeId, originalRig)
      }
      // Restaurer les collapses — utiliser forceUncollapse, PAS recoverNode.
      // recoverNode appelle recoverFromCollapse (dégradation permanente × 0.7) ce qui
      // corrompt le graphe à travers toutes les simulations PIR et cause un crash
      // par feedback runaway (breakingPoint → 0 → tout collapse → tout se dégrade).
      for (const nodeId of this.graph.getCollapsedNodes()) {
        if (!collapsedBefore.has(nodeId)) {
          this.graph.forceUncollapse(nodeId)
        }
      }
    }

    // Confidence : heuristique basée sur la proportion de nœuds stables
    const totalNodes    = this.graph.getAllNodes().length
    const collapseRatio = result!.collapseEvents.length / Math.max(totalNodes, 1)
    const confidence    = clamp01(1 - collapseRatio * 2)

    return {
      event,
      predictedDeltas:    result.beliefDeltas,
      predictedEntropy:   result.entropyAfter,
      predictedCollapses: result.collapseEvents,
      confidence,
    }
  }

  // ── Séquence de pressions ────────────────────────────────────────────

  applyPressureSequence(events: PressureEvent[], evidence: Evidence[] = []): PressureResult[] {
    return events.map(event => this.applyPressure(event, evidence))
  }

  // ── Accès au log ──────────────────────────────────────────────────────

  getPressureLog(): PressureEvent[] {
    return [...this.pressureLog]
  }

  getPressureCount(): number {
    return this.pressureLog.length
  }

  // ── Reset : restaure les priors originaux et réinitialise la rigidité ─

  reset(): void {
    // Restaurer les priors d'origine
    for (const [nodeId, originalPrior] of this.basePriors) {
      const node = this.graph.getNode(nodeId)
      if (node) node.prior = originalPrior
    }

    // Réinitialiser la rigidité pour chaque nœud
    for (const node of this.graph.getAllNodes()) {
      const freshState = initializeRigidity(node)
      this.graph.setRigidity(node.id, freshState)
      if (node.collapsed) {
        node.collapsed = false
      }
    }

    // Vider le byCollapsed index (via recoverNode pour chaque collapsed)
    for (const nodeId of [...this.graph.getCollapsedNodes()]) {
      this.graph.recoverNode(nodeId)
    }

    this.pressureLog.length = 0
  }

  // ── Utilitaire interne (apply sans push dans le log) ─────────────────

  private _applyWithoutLog(event: PressureEvent, evidence: Evidence[]): PressureResult {
    const savedLogLength = this.pressureLog.length
    const result = this.applyPressure(event, evidence)
    // retirer l'event du log (pop)
    this.pressureLog.splice(savedLogLength)
    return result
  }
}
