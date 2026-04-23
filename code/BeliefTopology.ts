// ============================================================
// EIDOLON Belief Topology Engine (TDA Layer)
// 
//
// RÔLE
// ════
// Analyser la STRUCTURE TOPOLOGIQUE du graphe d'affinité —
// pas les valeurs de croyance individuelles, mais leur
// organisation géométrique.
//
// HOMOLOGIE PERSISTANTE (H0 + H1)
// ════════════════════════════════
// Filtration descendante sur les poids d'affinité :
//   τ = 1.0 → 0.0 : on ajoute progressivement les arêtes
//
// H0 — Composantes connexes :
//   À τ=1, chaque nœud est sa propre composante (naissance=1).
//   Quand deux composantes fusionnent à τ=w : la plus jeune
//   meurt, persistence = τ_naissance - τ_mort.
//   Forte persistence → cluster structurellement stable.
//
// H1 — Cycles :
//   Un cycle dans le graphe d'affinité (à τ≥0.30) = cluster
//   de renforcement mutuel. Les nœuds en cycle s'influencent
//   circulairement → incompressibilité structurelle.
//
// CLASSES TOPOLOGIQUES
// ════════════════════
// Chaque nœud reçoit une classe selon sa position dans la
// structure persistante :
//
//   core      : persistence > 0.6 ET dans un cycle
//               → ancrage structural profond
//
//   bridge    : nœud-pont (sa suppression déconnecte le graphe)
//               → point de passage critique, haute vulnérabilité
//
//   boundary  : persistence > 0.25, pas de cycle
//               → nœud périphérique mais connecté
//
//   isolated  : faible persistence, peu de connexions
//               → dépendance superficielle
//
// ENTROPIE STRUCTURELLE
// ═════════════════════
// H_struct = -Σ p_i log2(p_i)  où p_i = pers_i / Σ pers
// Mesure la diversité topologique du graphe.
// Haute : structure complexe, diffuse.
// Basse  : structure concentrée, quelques nœuds dominants.
//
// INVARIANTS
// ══════════
// - Pur : aucune écriture, aucun effet de bord
// - Déterministe : même input → même output
// - Stable pour N=1 et graphes sans arêtes
// ============================================================

import type { AffinityMatrix } from './SemanticAffinity'

// ── Types exportés ────────────────────────────────────────────────────────────

export type TopologicalClass = 'core' | 'bridge' | 'boundary' | 'isolated'

export interface TopologicalProfile {
  nodeId:              string
  topologicalClass:    TopologicalClass
  persistence:         number   // [0,1] — durée de vie de la composante
  componentId:         number   // ID de la composante survivante
  inCycle:             boolean  // participates in a mutual-reinforcement loop
  isBridgeNode:        boolean  // removal increases component count
  coreScore:           number   // [0,∞) — score de centralité structurelle
}

export interface PersistenceComponent {
  id:          number
  nodes:       string[]
  persistence: number   // τ_born - τ_died
  bornAt:      number   // threshold where component appeared
  diedAt:      number   // threshold where component merged (0 if survivor)
  survived:    boolean  // never merged — highest persistence class
}

export interface DetectedCycle {
  nodes:       string[]  // nœuds participant au cycle
  minWeight:   number    // poids minimal dans le cycle (cycle persistence)
  meanBelief:  number    // croyance moyenne des nœuds du cycle
}

export interface TopologyResult {
  profiles:           Map<string, TopologicalProfile>
  components:         PersistenceComponent[]
  cycles:             DetectedCycle[]
  bottleneckNodes:    string[]  // bridge nodes with persistence > 0.3
  structuralEntropy:  number
  nodeCount:          number
}

// ── Constantes ────────────────────────────────────────────────────────────────
const CYCLE_DETECTION_THRESHOLD = 0.28   // seuil pour considérer un edge dans H1
const CORE_PERSISTENCE_MIN      = 0.55   // persistence minimale pour 'core'
const BRIDGE_PERSISTENCE_MIN    = 0.30   // persistence minimale pour bottleneck
const QUANTIZE_STEP             = 0.05   // quantification des seuils de filtration

// ── Union-Find avec tracking des membres ─────────────────────────────────────
class UnionFind {
  private parent  = new Map<string, string>()
  private rank    = new Map<string, number>()
  private members = new Map<string, Set<string>>()

  constructor(ids: string[]) {
    for (const id of ids) {
      this.parent.set(id, id)
      this.rank.set(id, 0)
      this.members.set(id, new Set([id]))
    }
  }

  find(id: string): string {
    const p = this.parent.get(id)
    if (!p) return id
    if (p !== id) {
      this.parent.set(id, this.find(p))
    }
    return this.parent.get(id)!
  }

  union(a: string, b: string): { survivor: string; merged: string } {
    const ra = this.find(a)
    const rb = this.find(b)
    if (ra === rb) return { survivor: ra, merged: ra }

    const rankA = this.rank.get(ra) ?? 0
    const rankB = this.rank.get(rb) ?? 0

    let survivor: string, merged: string
    if (rankA >= rankB) { survivor = ra; merged = rb }
    else                { survivor = rb; merged = ra }

    this.parent.set(merged, survivor)

    const survivorMembers = this.members.get(survivor)!
    const mergedMembers   = this.members.get(merged)!
    for (const m of mergedMembers) survivorMembers.add(m)
    this.members.set(merged, new Set())  // vider le merged

    if (rankA === rankB) this.rank.set(survivor, rankA + 1)

    return { survivor, merged }
  }

  getMembers(id: string): string[] {
    return [...(this.members.get(this.find(id)) ?? new Set())]
  }

  isConnected(a: string, b: string): boolean {
    return this.find(a) === this.find(b)
  }
}

// ── H0 : Homologie des composantes connexes ───────────────────────────────────
function computeH0(
  nodeIds: string[],
  matrix:  AffinityMatrix,
): { components: PersistenceComponent[]; componentMap: Map<string, number> } {
  if (nodeIds.length === 0) return { components: [], componentMap: new Map() }

  // Collecter toutes les arêtes et les trier par poids décroissant
  const edges: [string, string, number][] = []
  const seen = new Set<string>()
  for (const id of nodeIds) {
    const row = matrix.get(id)
    if (!row) continue
    for (const [nb, w] of row) {
      const key = [id, nb].sort().join('|')
      if (!seen.has(key) && nodeIds.includes(nb)) {
        seen.add(key)
        edges.push([id, nb, Math.round(w / QUANTIZE_STEP) * QUANTIZE_STEP])
      }
    }
  }
  edges.sort((a, b) => b[2] - a[2])

  const uf = new UnionFind(nodeIds)
  const components: PersistenceComponent[] = []
  const componentBirth = new Map<string, number>()
  let compIdCounter = 0

  // Chaque nœud naît comme sa propre composante à τ=1
  for (const id of nodeIds) {
    componentBirth.set(id, 1.0)
  }

  for (const [a, b, w] of edges) {
    if (uf.isConnected(a, b)) continue

    const rootA = uf.find(a)
    const rootB = uf.find(b)
    const bornA = componentBirth.get(rootA) ?? 1.0
    const bornB = componentBirth.get(rootB) ?? 1.0

    // La composante née plus récemment (bornAt plus petit = apparue à τ plus bas) meurt
    const [dying, surviving] = bornA <= bornB
      ? [rootA, rootB]
      : [rootB, rootA]

    const dyingBorn = componentBirth.get(dying) ?? 1.0
    const persistence = Math.max(0, dyingBorn - w)

    components.push({
      id:          compIdCounter++,
      nodes:       uf.getMembers(dying),
      persistence,
      bornAt:      dyingBorn,
      diedAt:      w,
      survived:    false,
    })

    const { survivor } = uf.union(a, b)
    // Le survivant naît à la plus ancienne naissance
    componentBirth.set(survivor, Math.max(bornA, bornB))
  }

  // Composantes survivantes (jamais fusionnées) — persistence maximale
  const survivingRoots = new Set(nodeIds.map(id => uf.find(id)))
  for (const root of survivingRoots) {
    const bornAt = componentBirth.get(root) ?? 1.0
    components.push({
      id:          compIdCounter++,
      nodes:       uf.getMembers(root),
      persistence: bornAt,
      bornAt,
      diedAt:      0,
      survived:    true,
    })
  }

  // Map nœud → composante (ID de la composante survivante qui le contient)
  const componentMap = new Map<string, number>()
  for (const comp of components) {
    if (comp.survived) {
      for (const nodeId of comp.nodes) {
        componentMap.set(nodeId, comp.id)
      }
    }
  }
  // Fallback pour les nœuds dans des composantes mortes : trouver leur survivant
  for (const id of nodeIds) {
    if (!componentMap.has(id)) {
      const root = uf.find(id)
      const survComp = components.find(c => c.survived && c.nodes.includes(root))
      componentMap.set(id, survComp?.id ?? -1)
    }
  }

  return { components, componentMap }
}

// ── H1 : Cycles via base de cycles du spanning tree ──────────────────────────
//
// Approche : Kruskal (max spanning tree) sur le graphe filtré.
// Les arêtes rejetées (nœuds déjà connectés) définissent exactement
// les cycles fondamentaux — une base canonique, sans faux positifs.
// Complexité : O(E log E + V·cycles) — stable sur graphes denses.
function detectCyclesBasis(
  nodeIds: string[],
  matrix:  AffinityMatrix,
  beliefs: Map<string, number>,
): DetectedCycle[] {
  const nodeSet = new Set(nodeIds)

  // Collecter les arêtes au-dessus du seuil
  const edges: [string, string, number][] = []
  const seen = new Set<string>()
  for (const id of nodeIds) {
    const row = matrix.get(id)
    if (!row) continue
    for (const [nb, w] of row) {
      if (w < CYCLE_DETECTION_THRESHOLD || !nodeSet.has(nb)) continue
      const key = id < nb ? `${id}|${nb}` : `${nb}|${id}`
      if (!seen.has(key)) { seen.add(key); edges.push([id, nb, w]) }
    }
  }
  edges.sort((a, b) => b[2] - a[2])  // décroissant : arêtes fortes en premier

  // MST max via Union-Find + tracking des arêtes de cycle
  const uf = new UnionFind(nodeIds)
  const treeAdj = new Map<string, Map<string, number>>()
  for (const id of nodeIds) treeAdj.set(id, new Map())

  const cycleEdges: [string, string, number][] = []
  for (const [a, b, w] of edges) {
    if (uf.isConnected(a, b)) {
      cycleEdges.push([a, b, w])   // arête de cycle fondamental
    } else {
      uf.union(a, b)
      treeAdj.get(a)!.set(b, w)
      treeAdj.get(b)!.set(a, w)
    }
  }

  // Reconstruire chaque cycle fondamental par BFS dans le MST
  const cycles: DetectedCycle[] = []
  const cycleSignatures = new Set<string>()

  for (const [a, b, wEdge] of cycleEdges) {
    const path = bfsTreePath(a, b, treeAdj, nodeSet)
    if (!path || path.length < 3) continue

    const sig = [...path].sort().join('|')
    if (cycleSignatures.has(sig)) continue
    cycleSignatures.add(sig)

    // Poids minimal : min(arêtes MST du chemin, arête de fermeture)
    let minWeight = wEdge
    for (let i = 0; i < path.length - 1; i++) {
      minWeight = Math.min(minWeight, treeAdj.get(path[i])?.get(path[i + 1]) ?? 0)
    }

    const meanBelief = path.reduce((s, id) => s + (beliefs.get(id) ?? 0.2), 0) / path.length
    cycles.push({ nodes: path, minWeight, meanBelief })
  }

  return cycles
}

// BFS dans l'arbre de spanning pour trouver le chemin de src à dst
function bfsTreePath(
  src:     string,
  dst:     string,
  treeAdj: Map<string, Map<string, number>>,
  nodeSet: Set<string>,
): string[] | null {
  if (src === dst) return [src]
  const visited = new Set<string>([src])
  const prev    = new Map<string, string>()
  const queue   = [src]

  while (queue.length > 0) {
    const cur = queue.shift()!
    if (cur === dst) {
      const path: string[] = [dst]
      let c = dst
      while (c !== src) {
        c = prev.get(c)!
        path.unshift(c)
      }
      return path
    }
    for (const [nb] of treeAdj.get(cur) ?? new Map()) {
      if (!visited.has(nb) && nodeSet.has(nb)) {
        visited.add(nb)
        prev.set(nb, cur)
        queue.push(nb)
      }
    }
  }
  return null  // pas de chemin dans l'arbre (composantes séparées)
}

// ── Nœuds d'articulation : algorithme de Tarjan, O(V+E) — sans cap de taille ──
// Un nœud d'articulation est un nœud dont la suppression augmente
// le nombre de composantes connexes du graphe filtré.
function computeArticulationPoints(
  nodeIds:   string[],
  matrix:    AffinityMatrix,
  threshold: number,
): Set<string> {
  const nodeSet = new Set(nodeIds)
  const disc    = new Map<string, number>()  // temps de découverte
  const low     = new Map<string, number>()  // plus petit disc atteignable
  const articulations = new Set<string>()
  let timer = 0

  const dfs = (node: string, parentId: string | null): void => {
    disc.set(node, timer)
    low.set(node, timer)
    timer++
    let childCount = 0

    const row = matrix.get(node)
    if (!row) return

    for (const [nb, w] of row) {
      if (w < threshold || !nodeSet.has(nb)) continue
      if (!disc.has(nb)) {
        childCount++
        dfs(nb, node)
        low.set(node, Math.min(low.get(node)!, low.get(nb)!))
        // Racine avec 2+ enfants → point d'articulation
        if (parentId === null && childCount > 1)              articulations.add(node)
        // Nœud non-racine : si aucun descendant ne peut remonter plus haut
        if (parentId !== null && low.get(nb)! >= disc.get(node)!) articulations.add(node)
      } else if (nb !== parentId) {
        // Arête de retour : mettre à jour low
        low.set(node, Math.min(low.get(node)!, disc.get(nb)!))
      }
    }
  }

  for (const id of nodeIds) {
    if (!disc.has(id)) dfs(id, null)
  }

  return articulations
}

// ── Entropie structurelle du diagramme de persistence ────────────────────────
function structuralEntropy(components: PersistenceComponent[]): number {
  const total = components.reduce((s, c) => s + c.persistence, 0)
  if (total < 1e-10) return 0
  return -components
    .filter(c => c.persistence > 0)
    .reduce((s, c) => {
      const p = c.persistence / total
      return s + p * Math.log2(p + 1e-10)
    }, 0)
}

// ── Classe topologique d'un nœud ─────────────────────────────────────────────
function classifyNode(
  nodeId:      string,
  persistence: number,
  inCycle:     boolean,
  isBridge:    boolean,
): TopologicalClass {
  if (persistence >= CORE_PERSISTENCE_MIN && inCycle) return 'core'
  if (isBridge)                                        return 'bridge'
  if (persistence > 0.25 || inCycle)                  return 'boundary'
  return 'isolated'
}

// ── Point d'entrée principal ──────────────────────────────────────────────────

/**
 * Calcule la topologie persistante du graphe d'affinité.
 *
 * @param canvasNodeIds  Nœuds présents sur le canvas
 * @param matrix         Matrice d'affinité (de SemanticAffinity)
 * @param beliefs        Croyances courantes (de BeliefPropagation)
 * @returns              Profils topologiques + diagramme de persistence
 */
export function computeTopology(
  canvasNodeIds: string[],
  matrix:        AffinityMatrix,
  beliefs:       Map<string, number>,
): TopologyResult {
  const n = canvasNodeIds.length

  if (n === 0) {
    return {
      profiles: new Map(), components: [], cycles: [],
      bottleneckNodes: [], structuralEntropy: 0, nodeCount: 0,
    }
  }

  if (n === 1) {
    const id = canvasNodeIds[0]
    return {
      profiles: new Map([[id, {
        nodeId: id, topologicalClass: 'isolated', persistence: 1.0,
        componentId: 0, inCycle: false, isBridgeNode: false, coreScore: 0,
      }]]),
      components: [{ id: 0, nodes: [id], persistence: 1.0, bornAt: 1.0, diedAt: 0, survived: true }],
      cycles: [], bottleneckNodes: [], structuralEntropy: 0, nodeCount: 1,
    }
  }

  // ── H0 : composantes connexes ─────────────────────────────────────────────
  const { components, componentMap } = computeH0(canvasNodeIds, matrix)

  // ── H1 : cycles (base de cycles, stable sur graphes denses) ─────────────
  const cycles = detectCyclesBasis(canvasNodeIds, matrix, beliefs)
  const nodesInCycle = new Set(cycles.flatMap(c => c.nodes))

  // ── Nœuds d'articulation (Tarjan O(V+E), pas de cap de taille) ───────────
  const bridgeNodes = computeArticulationPoints(canvasNodeIds, matrix, CYCLE_DETECTION_THRESHOLD)

  // ── Persistence par nœud ──────────────────────────────────────────────────
  // Nœud hérite de la persistence de la composante qui le contient au moment de sa mort
  const nodePersistence = new Map<string, number>()
  for (const comp of components) {
    for (const id of comp.nodes) {
      const cur = nodePersistence.get(id) ?? 0
      nodePersistence.set(id, Math.max(cur, comp.persistence))
    }
  }
  // Borne explicite [0,1] — défense contre toute dérive de l'AffinityMatrix amont
  for (const [id, pers] of nodePersistence) {
    nodePersistence.set(id, Math.max(0, Math.min(1, pers)))
  }

  // ── Profils par nœud ──────────────────────────────────────────────────────
  const profiles = new Map<string, TopologicalProfile>()
  for (const nodeId of canvasNodeIds) {
    const pers    = nodePersistence.get(nodeId) ?? 0
    const inCycle = nodesInCycle.has(nodeId)
    const isBrid  = bridgeNodes.has(nodeId)
    const compId  = componentMap.get(nodeId) ?? -1
    const cls     = classifyNode(nodeId, pers, inCycle, isBrid)
    const core    = pers * (inCycle ? 1.6 : 1.0) * (isBrid ? 1.3 : 1.0)

    profiles.set(nodeId, {
      nodeId,
      topologicalClass: cls,
      persistence:      pers,
      componentId:      compId,
      inCycle,
      isBridgeNode:     isBrid,
      coreScore:        core,
    })
  }

  // ── Bottleneck nodes ──────────────────────────────────────────────────────
  const bottleneckNodes = canvasNodeIds.filter(id => {
    const p = profiles.get(id)!
    return p.isBridgeNode && p.persistence >= BRIDGE_PERSISTENCE_MIN
  })

  return {
    profiles,
    components,
    cycles,
    bottleneckNodes,
    structuralEntropy: structuralEntropy(components),
    nodeCount: n,
  }
}

// ── Utilitaire : score de disruption topologique ─────────────────────────────
/**
 * Calcule à quel point cibler nodeId disrupte les cycles persistants.
 * Utilisé par le PIR Solver pour l'objectif étendu.
 */
export function topologicalDisruptionScore(
  nodeId:   string,
  topology: TopologyResult,
  matrix:   AffinityMatrix,
): number {
  if (topology.cycles.length === 0) return 0

  let score = 0
  const row = matrix.get(nodeId)

  for (const cycle of topology.cycles) {
    // Affinité du candidat vers les nœuds du cycle
    let affinityToCycle = 0
    for (const cn of cycle.nodes) {
      if (cn === nodeId) {
        // Le nœud EST dans le cycle → disruption directe maximale
        affinityToCycle += 1.0
      } else {
        affinityToCycle += row?.get(cn) ?? 0
      }
    }
    affinityToCycle /= cycle.nodes.length

    // Score = persistence du cycle × affinité → disruption potentielle
    score += cycle.minWeight * affinityToCycle
  }

  // Normaliser par le nombre de cycles
  return score / topology.cycles.length
}
