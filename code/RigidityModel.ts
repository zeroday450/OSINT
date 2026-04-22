// ============================================================
// EIDOLON Rigidity Model
// 
//
// MODULE PUREMENT MATHÉMATIQUE. Aucun LLM. Aucun état React.
//
// DÉFINITION FORMELLE
// ═══════════════════
//
// Chaque noeud v a un état de rigidité RigidityState qui modélise
// sa résistance à la déformation sous pression externe.
//
// Paramètres :
//   rigidity       r ∈ [0,1]  — résistance au changement de belief
//                               0 = fluide (pression passe à 100%)
//                               1 = rigide absolu (pression bloquée)
//   elasticity     e ∈ [0,1]  — capacité de retour à l'état initial
//                               0 = déformation permanente
//                               1 = retour complet (ressort parfait)
//   breakingPoint  β ∈ [0,1]  — seuil de strain cumulé pour collapse
//   currentStrain  σ ∈ [0,+∞) — pression cumulée non résolue
//
// Invariant : prior ⊥ rigidity
//   prior = taux de base dans la population [0,1]
//   rigidity = résistance à la pression externe [0,1]
//   Ce sont deux axes ORTHOGONAUX. Ne jamais les confondre.
//
// MÉCANIQUE DE DÉFORMATION
// ════════════════════════
//
//   effectivePressure = pressureMagnitude × (1 - rigidity)
//   newStrain = currentStrain + effectivePressure
//   if newStrain ≥ breakingPoint → collapse
//
// WORK HARDENING
// ══════════════
//   La résistance augmente avec l'historique de déformation
//   (comme un métal qui durcit sous contrainte répétée).
//   historyFactor = mean(deformationHistory)
//   cost = rigidity × (1 + historyFactor) × pressureMagnitude
//
// RELAXATION EXPONENTIELLE
// ════════════════════════
//   strain(t+dt) = strain(t) × exp(-dt / τ)
//   τ = 1 / (elasticity + ε)
//   Haute élasticité → relaxation rapide.
//
// COLLAPSE ET RÉCUPÉRATION
// ════════════════════════
//   Après collapse : nœud ne résiste plus (collapsed=true).
//   Récupération possible mais avec perte permanente :
//     rigidity  *= 0.7    (brisé, moins résistant)
//     elasticity *= 0.8   (moins élastique)
//     breakingPoint *= 0.9 (seuil abaissé — plus facile à recasser)
//
// ============================================================

import type { GraphNode, GraphEdge } from '../types'

// ── État de rigidité d'un nœud ─────────────────────────────────────

export interface RigidityState {
  nodeId:            string
  rigidity:          number    // [0,1] — résistance au changement
  elasticity:        number    // [0,1] — capacité de retour à baseline
  breakingPoint:     number    // [0,1] — seuil de collapse
  currentStrain:     number    // [0,+∞) — pression cumulée non résolue
  deformationHistory: number[] // historique des effectivePressure subis
  collapsed:         boolean
}

// ── Initialisation ─────────────────────────────────────────────────

/**
 * Initialise l'état de rigidité d'un noeud depuis ses paramètres GraphNode.
 * Utilise les defaultRigidity/Elasticity/BreakingPoint si les champs Phase 2
 * ne sont pas encore setés explicitement.
 */
export function initializeRigidity(node: GraphNode): RigidityState {
  const props = (node.properties ?? {}) as Record<string, number>
  return {
    nodeId:             node.id,
    rigidity:           node.rigidity          ?? props.defaultRigidity      ?? 0.5,
    elasticity:         node.elasticity        ?? props.defaultElasticity    ?? 0.5,
    breakingPoint:      node.breakingPoint     ?? props.defaultBreakingPoint ?? 0.9,
    currentStrain:      0,
    deformationHistory: [],
    collapsed:          node.collapsed ?? false,
  }
}

// ── Coût de déformation ────────────────────────────────────────────

/**
 * Coût de déformation = énergie nécessaire pour que la pression donnée
 * produise un changement de belief effectif.
 *
 * Formule : cost = rigidity × (1 + historyFactor) × pressureMagnitude
 *
 * Work hardening : plus un nœud a été déformé, plus il résiste.
 * historyFactor = moyenne des déformations précédentes.
 */
export function computeDeformationCost(
  state: RigidityState,
  pressureMagnitude: number
): number {
  const historyFactor = state.deformationHistory.length > 0
    ? state.deformationHistory.reduce((a, b) => a + b, 0) / state.deformationHistory.length
    : 0
  return state.rigidity * (1 + historyFactor) * pressureMagnitude
}

// ── Application de contrainte (strain) ────────────────────────────

/**
 * Applique une contrainte sur le nœud. Retourne le nouvel état (immuable).
 *
 * effectivePressure = pressureMagnitude × (1 - rigidity)
 * newStrain = currentStrain + effectivePressure
 * Si newStrain ≥ breakingPoint → collapse.
 * Sinon → relaxation élastique partielle.
 */
export function applyStrain(
  state: RigidityState,
  pressureMagnitude: number
): RigidityState {
  if (state.collapsed) return state  // un nœud collapsed n'accumule plus de strain

  const effectivePressure = pressureMagnitude * (1 - state.rigidity)
  const newStrain         = state.currentStrain + effectivePressure

  const newHistory = [...state.deformationHistory, effectivePressure]

  if (newStrain >= state.breakingPoint) {
    return {
      ...state,
      currentStrain:      newStrain,
      collapsed:          true,
      deformationHistory: newHistory,
    }
  }

  // Relaxation élastique partielle (10% par tick, modulée par élasticité)
  const relaxedStrain = newStrain * (1 - state.elasticity * 0.1)

  return {
    ...state,
    currentStrain:      relaxedStrain,
    deformationHistory: newHistory,
  }
}

// ── Shift de belief effectif après résistance ──────────────────────

/**
 * Étant donné un delta de belief brut (calculé par BP) et l'état de rigidité,
 * retourne le delta effectif après résistance.
 *
 * Un nœud très rigide laisse passer seulement une fraction du delta.
 * Un nœud collapsed n'offre plus aucune résistance.
 *
 * Formule : effectiveDelta = rawDelta × (1 - rigidity)
 * Cas collapse : effectiveDelta = rawDelta (plus de résistance)
 */
export function computeEffectivePriorShift(
  state: RigidityState,
  rawBeliefDelta: number
): number {
  if (state.collapsed) return rawBeliefDelta
  const resistanceFactor = 1 - state.rigidity
  return rawBeliefDelta * resistanceFactor
}

// ── Relaxation temporelle du strain ───────────────────────────────

/**
 * Relaxation exponentielle du strain au fil du temps.
 * τ = 1 / (elasticity + ε)  — haute élasticité → relaxation rapide
 * strain(t+dt) = strain(t) × exp(-dt / τ)
 *
 * @param dt — delta temps en unités arbitraires (1 = un tick standard)
 */
export function relaxStrain(state: RigidityState, dt: number): RigidityState {
  if (state.collapsed) return state
  const tau         = 1 / (state.elasticity + 1e-9)
  const decayFactor = Math.exp(-dt / tau)
  return {
    ...state,
    currentStrain: state.currentStrain * decayFactor,
  }
}

// ── Récupération après collapse ────────────────────────────────────

/**
 * Reset contrôlé après collapse. Le noeud redevient actif mais avec
 * des paramètres dégradés de façon permanente :
 *   rigidity    × 0.7   — brisé, moins résistant
 *   elasticity  × 0.8   — moins élastique
 *   breakingPoint × 0.9 — seuil abaissé (plus facile à recasser)
 *
 * Le nœud ne revient jamais complètement à son état initial.
 * C'est une décision architecturale délibérée : le collapse laisse
 * une trace permanente dans la structure cognitive de l'entité.
 */
export function recoverFromCollapse(state: RigidityState): RigidityState {
  return {
    ...state,
    collapsed:      false,
    currentStrain:  0,
    rigidity:       state.rigidity    * 0.7,
    elasticity:     state.elasticity  * 0.8,
    breakingPoint:  state.breakingPoint * 0.9,
  }
}

// ── Conductivité d'une arête ───────────────────────────────────────

/**
 * La conductivité d'une arête détermine quelle fraction de la pression
 * se propage le long de cette arête (Phase 3 — PressureEngine).
 *
 * Formule : conductivity = weight × (1 - avgRigidity) × edge.conductivity
 * avgRigidity = (source.rigidity + target.rigidity) / 2
 *
 * Plus les deux nœuds extrêmes sont rigides, moins la pression passe.
 * L'arête a aussi sa propre conductivité intrinsèque.
 */
export function computeEdgeConductivity(
  edge: GraphEdge,
  sourceRigidity: RigidityState,
  targetRigidity: RigidityState
): number {
  const avgRigidity = (sourceRigidity.rigidity + targetRigidity.rigidity) / 2
  const baseWeight  = edge.weight ?? 1.0
  return baseWeight * (1 - avgRigidity) * (edge.conductivity ?? 1.0)
}

// ── Utilitaire de diagnostic ───────────────────────────────────────

/**
 * Retourne le margin de sécurité avant collapse.
 * margin = breakingPoint - currentStrain
 * 0 = au bord du collapse. Négatif = déjà au-delà.
 */
export function collapseMargin(state: RigidityState): number {
  return state.breakingPoint - state.currentStrain
}

/**
 * Résistance observée = ratio pression_appliquée / |delta_belief|
 * Signal analytique clé : un ratio élevé = contrainte incompressible.
 * Undefined si delta_belief = 0 (nœud n'a pas bougé du tout).
 */
export function computeObservedResistance(
  pressureMagnitude: number,
  absoluteBeliefDelta: number
): number | undefined {
  if (absoluteBeliefDelta < 1e-9) return undefined  // nœud immobile — résistance infinie
  return pressureMagnitude / absoluteBeliefDelta
}
