// ============================================================
// EIDOLON Semantic Affinity Engine
// 
//
// RÔLE
// ════
// Calculer la matrice d'affinité N×N entre tous les nœuds
// canvas. Comme le mécanisme d'attention dans un transformer,
// chaque nœud "voit" tous les autres à travers cette matrice.
// L'utilisateur ne formalise pas les liaisons — le moteur
// les infère depuis :
//
//   W(i,j) = α·CatSim(i,j) + β·KBedge(i,j) + γ·TemporalCov(i,j)
//
// La matrice est symétrique, normalisée dans [0,1], et
// seuillée pour ne conserver que les connexions significatives.
//
// DOMAINE SÉMANTIQUE
// ══════════════════
// 18 catégories × 18 : affinités de domaine calibrées sur
// la structure cognitive OSINT. Chaque paire a une affinité
// de base reflétant leur co-occurrence dans les profils réels.
//
// COVARIANCE TEMPORELLE
// ═════════════════════
// Si les croyances de deux nœuds évoluent de façon corrélée
// sous pression (Pearson r > 0), ils partagent une influence
// structurelle — même sans edge KB explicite.
//
// INVARIANTS
// ══════════
// - Symétrique : W(i,j) = W(j,i)
// - Diagonale nulle : W(i,i) = 0
// - Seuil minimal : paires < AFFINITY_MIN_THRESHOLD ignorées
// - Pur : aucune écriture graphe, aucun effet de bord
// ============================================================

import type { GraphNode } from '../types'

// ── Seuil minimal d'affinité pour conserver une connexion ────────────────────
export const AFFINITY_MIN_THRESHOLD = 0.15

// ── Type exporté ─────────────────────────────────────────────────────────────
/** Matrice d'affinité symétrique : nodeId → (neighborId → weight ∈ [0,1]) */
export type AffinityMatrix = Map<string, Map<string, number>>

// ── Affinités de domaine inter-catégories ────────────────────────────────────
// Calibrées sur la structure sémantique OSINT.
// Asymétriques dans le lookup mais normalisées en max(a→b, b→a) à la lecture.
const DOMAIN_AFFINITY: Record<string, Record<string, number>> = {
  biological: {
    health: 0.90, psychological: 0.55, temporal: 0.40,
    geographic: 0.30, constraints: 0.35, vulnerabilities: 0.45,
  },
  geographic: {
    cultural: 0.55, temporal: 0.40, constraints: 0.45,
    economic: 0.40, linguistic: 0.35, biological: 0.30,
    social: 0.30,
  },
  temporal: {
    constraints: 0.60, economic: 0.45, biological: 0.40,
    geographic: 0.40, health: 0.45, professional: 0.45,
  },
  linguistic: {
    cultural: 0.80, social: 0.60, ideological: 0.50,
    geographic: 0.35, psychological: 0.40, behavioral: 0.45,
  },
  cultural: {
    social: 0.85, ideological: 0.80, behavioral: 0.65,
    linguistic: 0.80, motivations: 0.55, geographic: 0.55,
    interests: 0.50,
  },
  social: {
    behavioral: 0.80, psychological: 0.65, cultural: 0.85,
    motivations: 0.55, professional: 0.50, linguistic: 0.60,
    interests: 0.55, vulnerabilities: 0.45,
  },
  behavioral: {
    psychological: 0.85, motivations: 0.75, social: 0.80,
    approach: 0.70, vulnerabilities: 0.65, cultural: 0.65,
    interests: 0.65, constraints: 0.45,
  },
  interests: {
    motivations: 0.80, behavioral: 0.65, social: 0.55,
    professional: 0.50, digital: 0.45, cultural: 0.50,
    psychological: 0.55,
  },
  digital: {
    professional: 0.70, behavioral: 0.50, social: 0.50,
    constraints: 0.40, interests: 0.45, temporal: 0.30,
    economic: 0.40,
  },
  professional: {
    economic: 0.75, social: 0.55, motivations: 0.60,
    digital: 0.70, constraints: 0.55, temporal: 0.45,
    interests: 0.50,
  },
  economic: {
    constraints: 0.90, motivations: 0.60, vulnerabilities: 0.65,
    professional: 0.75, temporal: 0.45, geographic: 0.40,
    approach: 0.50,
  },
  constraints: {
    economic: 0.90, vulnerabilities: 0.75, motivations: 0.55,
    approach: 0.55, temporal: 0.60, psychological: 0.50,
    health: 0.55,
  },
  motivations: {
    psychological: 0.85, approach: 0.80, ideological: 0.75,
    behavioral: 0.75, vulnerabilities: 0.65, interests: 0.80,
    economic: 0.60, social: 0.55,
  },
  psychological: {
    behavioral: 0.85, motivations: 0.85, vulnerabilities: 0.80,
    ideological: 0.70, health: 0.65, social: 0.65,
    biological: 0.55, constraints: 0.50,
  },
  ideological: {
    cultural: 0.80, motivations: 0.75, psychological: 0.70,
    approach: 0.65, linguistic: 0.50, vulnerabilities: 0.60,
    social: 0.55,
  },
  health: {
    biological: 0.90, psychological: 0.65, constraints: 0.55,
    vulnerabilities: 0.60, temporal: 0.45, economic: 0.40,
  },
  vulnerabilities: {
    approach: 0.85, psychological: 0.80, constraints: 0.75,
    behavioral: 0.65, economic: 0.65, motivations: 0.65,
    health: 0.60, ideological: 0.60, social: 0.45,
  },
  approach: {
    vulnerabilities: 0.85, motivations: 0.80, behavioral: 0.70,
    ideological: 0.65, constraints: 0.55, economic: 0.50,
    psychological: 0.55,
  },
}

// ── Affinité de catégorie : max des deux directions (symétrie garantie) ───────
// Plancher non-nul : catégories connues mais sans affinité explicite → 0.08
// Catégories inconnues (hors des 18 domaines) → 0.05
// Garantit qu'aucun nœud n'est traité comme isolé par défaut de table.
const KNOWN_CATEGORIES = new Set(Object.keys(DOMAIN_AFFINITY))
const AFFINITY_FLOOR_KNOWN   = 0.08  // paire connue sans entrée explicite
const AFFINITY_FLOOR_UNKNOWN = 0.05  // catégorie hors table

function categorySimilarity(catA: string, catB: string): number {
  if (catA === catB) return 1.0
  const ab = DOMAIN_AFFINITY[catA]?.[catB] ?? 0
  const ba = DOMAIN_AFFINITY[catB]?.[catA] ?? 0
  const explicit = Math.max(ab, ba)
  if (explicit > 0) return explicit
  // Pas d'entrée explicite — appliquer le plancher approprié
  const bothKnown = KNOWN_CATEGORIES.has(catA) && KNOWN_CATEGORIES.has(catB)
  return bothKnown ? AFFINITY_FLOOR_KNOWN : AFFINITY_FLOOR_UNKNOWN
}

// ── Covariance de Pearson sur les historiques de croyance ────────────────────
// Prend les N dernières valeurs pour la stabilité numérique
function pearsonCorrelation(xs: number[], ys: number[], maxLen = 12): number {
  const n = Math.min(xs.length, ys.length, maxLen)
  if (n < 3) return 0

  const xSlice = xs.slice(-n)
  const ySlice = ys.slice(-n)

  const mx = xSlice.reduce((a, b) => a + b, 0) / n
  const my = ySlice.reduce((a, b) => a + b, 0) / n

  let num = 0, dx2 = 0, dy2 = 0
  for (let i = 0; i < n; i++) {
    const dx = xSlice[i] - mx
    const dy = ySlice[i] - my
    num  += dx * dy
    dx2  += dx * dx
    dy2  += dy * dy
  }
  const denom = Math.sqrt(dx2 * dy2)
  if (denom < 1e-10) return 0
  return Math.max(-1, Math.min(1, num / denom))
}

// ── Interface du graphe — uniquement ce dont on a besoin ─────────────────────
interface GraphLike {
  getOutgoingEdges: (id: string) => Array<{ target: string; weight: number }>
  getIncomingEdges: (id: string) => Array<{ source: string; weight: number }>
  getNode:          (id: string) => GraphNode | undefined
}

// ── Calcul de la paire (i, j) ────────────────────────────────────────────────
function pairAffinity(
  nodeA:       GraphNode,
  nodeB:       GraphNode,
  kbEdgeScore: number,          // [0,1] force de l'edge KB combiné A↔B
  temporalCov: number,          // [-1,1] corrélation Pearson des croyances
): number {
  // Poids des composantes :
  //   40% catégorie   — connaissance domaine stable
  //   35% edge KB     — relation explicite connue
  //   25% covariance  — relation inférée par comportement observé
  const catSim  = categorySimilarity(nodeA.category, nodeB.category)
  const covNorm = Math.max(0, temporalCov)  // seule corrélation positive compte

  const raw = 0.40 * catSim + 0.35 * kbEdgeScore + 0.25 * covNorm

  // Bonus de sous-catégorie : même sous-catégorie → +0.10
  const subBonus = (nodeA.subcategory && nodeB.subcategory &&
    nodeA.subcategory === nodeB.subcategory) ? 0.10 : 0

  return Math.min(1, raw + subBonus)
}

// ── Point d'entrée principal ──────────────────────────────────────────────────

/**
 * Calcule la matrice d'affinité sémantique N×N pour les nœuds canvas.
 *
 * @param canvasNodeIds  IDs des nœuds présents sur le canvas
 * @param graph          Graphe de connaissances (pour les edges KB)
 * @param beliefHistory  Historique des croyances par nœud (de TemporalLayer)
 * @returns              Matrice symétrique : id → (id → weight ∈ [0,1])
 */
export function computeAffinityMatrix(
  canvasNodeIds: string[],
  graph:         GraphLike,
  beliefHistory: Map<string, number[]> = new Map(),
): AffinityMatrix {
  const matrix: AffinityMatrix = new Map()

  // Initialiser toutes les lignes
  for (const id of canvasNodeIds) {
    matrix.set(id, new Map())
  }

  const n = canvasNodeIds.length

  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const idA  = canvasNodeIds[i]
      const idB  = canvasNodeIds[j]
      const nodeA = graph.getNode(idA)
      const nodeB = graph.getNode(idB)
      if (!nodeA || !nodeB) continue

      // ── KB edge score : max(A→B, B→A) ────────────────────────────────────
      const outA   = graph.getOutgoingEdges(idA)
      const edgeAB = outA.find(e => e.target === idB)
      const outB   = graph.getOutgoingEdges(idB)
      const edgeBA = outB.find(e => e.target === idA)
      const kbScore = Math.min(1, Math.max(edgeAB?.weight ?? 0, edgeBA?.weight ?? 0))

      // ── Covariance temporelle ─────────────────────────────────────────────
      const histA = beliefHistory.get(idA) ?? []
      const histB = beliefHistory.get(idB) ?? []
      const cov   = pearsonCorrelation(histA, histB)

      // ── Affinité combinée ─────────────────────────────────────────────────
      const w = pairAffinity(nodeA, nodeB, kbScore, cov)

      if (w >= AFFINITY_MIN_THRESHOLD) {
        matrix.get(idA)!.set(idB, w)
        matrix.get(idB)!.set(idA, w)
      }
    }
  }

  return matrix
}

// ── Utilitaire : affinité d'un nœud vers un ensemble ─────────────────────────
/**
 * Affinité moyenne d'un nœud source vers un ensemble de nœuds cibles.
 * Utile pour le PIR Solver : score topologique d'un candidat vs un cycle.
 */
export function nodeSetAffinity(
  sourceId:  string,
  targetIds: string[],
  matrix:    AffinityMatrix,
): number {
  if (targetIds.length === 0) return 0
  const row = matrix.get(sourceId)
  if (!row || row.size === 0) return 0

  let sum = 0
  let count = 0
  for (const tid of targetIds) {
    if (tid === sourceId) continue
    const w = row.get(tid) ?? 0
    sum += w
    count++
  }
  return count > 0 ? sum / count : 0
}
