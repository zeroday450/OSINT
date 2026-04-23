// ============================================================
// EIDOLON Pressure Generator
// 
//
// Génère des PressureEvents valides, typés, et horodatés.
// Toutes les fonctions retournent des objets immuables.
//
// Conventions :
//   magnitude ∈ [0, 1]   — 0.3 = légère, 0.6 = modérée, 0.9 = intense
//   profile             — forme temporelle (impulse par défaut)
//   source              — origine de l'event (manual par défaut)
// ============================================================

import {
  type PressureEvent,
  type PressureVector,
  PressureDirection,
  PressureProfile,
  PressureSource,
} from '../types/pressure'

let _eventCounter = 0

function nextId(prefix = 'pe'): string {
  return `${prefix}_${Date.now()}_${++_eventCounter}`
}

function makeVector(
  direction: PressureDirection,
  magnitude: number,
  profile: PressureProfile,
): PressureVector {
  return {
    magnitude: Math.max(0, magnitude),
    direction,
    profile,
  }
}

// ── Factories sémantiques ──────────────────────────────────────

/**
 * Pression de CONFIRMATION — renforce le belief actuel.
 * Pousse le prior vers 1 : prior += magnitude × (1 - prior)
 * Cas d'usage : valider une hypothèse, renforcer une conviction.
 */
export function createConfirmPressure(
  targetNodeIds: string[],
  magnitude: number,
  profile: PressureProfile = PressureProfile.impulse,
  source: PressureSource = PressureSource.manual,
  metadata: Record<string, unknown> = {},
): PressureEvent {
  return {
    id:            nextId('confirm'),
    timestamp:     Date.now(),
    targetNodeIds: [...targetNodeIds],
    pressureVector: makeVector(PressureDirection.confirm, magnitude, profile),
    source,
    metadata,
  }
}

/**
 * Pression de CONTRADICTION — contredit le belief actuel.
 * Pousse le prior vers 0 : prior -= magnitude × prior
 * Cas d'usage : confronter avec une preuve contraire, nier une assomption.
 */
export function createContradictPressure(
  targetNodeIds: string[],
  magnitude: number,
  profile: PressureProfile = PressureProfile.impulse,
  source: PressureSource = PressureSource.manual,
  metadata: Record<string, unknown> = {},
): PressureEvent {
  return {
    id:            nextId('contradict'),
    timestamp:     Date.now(),
    targetNodeIds: [...targetNodeIds],
    pressureVector: makeVector(PressureDirection.contradict, magnitude, profile),
    source,
    metadata,
  }
}

/**
 * Pression d'AMBIGUÏTÉ — introduit de l'incertitude.
 * Pousse le prior vers 0.5 : prior → 0.5 + (prior - 0.5) × (1 - magnitude)
 * Cas d'usage : brouiller une conviction, tester l'élasticité d'une croyance.
 */
export function createAmbiguatePressure(
  targetNodeIds: string[],
  magnitude: number,
  profile: PressureProfile = PressureProfile.sustained,
  source: PressureSource = PressureSource.manual,
  metadata: Record<string, unknown> = {},
): PressureEvent {
  return {
    id:            nextId('ambiguate'),
    timestamp:     Date.now(),
    targetNodeIds: [...targetNodeIds],
    pressureVector: makeVector(PressureDirection.ambiguate, magnitude, profile),
    source,
    metadata,
  }
}

/**
 * Pression de POLARISATION — pousse vers 0 ou 1 selon le prior dominant.
 * Si prior > 0.5 → push vers 1, si prior < 0.5 → push vers 0.
 * Cas d'usage : forcer une décision binaire, révéler la conviction profonde.
 */
export function createPolarizePressure(
  targetNodeIds: string[],
  magnitude: number,
  profile: PressureProfile = PressureProfile.impulse,
  source: PressureSource = PressureSource.manual,
  metadata: Record<string, unknown> = {},
): PressureEvent {
  return {
    id:            nextId('polarize'),
    timestamp:     Date.now(),
    targetNodeIds: [...targetNodeIds],
    pressureVector: makeVector(PressureDirection.polarize, magnitude, profile),
    source,
    metadata,
  }
}

/**
 * Factory générique — crée un PressureEvent avec tous les paramètres.
 */
export function createPressure(params: {
  targetNodeIds: string[]
  direction:     PressureDirection
  magnitude:     number
  profile?:      PressureProfile
  source?:       PressureSource
  metadata?:     Record<string, unknown>
}): PressureEvent {
  return {
    id:            nextId('pe'),
    timestamp:     Date.now(),
    targetNodeIds: [...params.targetNodeIds],
    pressureVector: makeVector(
      params.direction,
      params.magnitude,
      params.profile ?? PressureProfile.impulse,
    ),
    source:   params.source   ?? PressureSource.manual,
    metadata: params.metadata ?? {},
  }
}

/**
 * Génère un event PIR — taggé `source: solver`.
 * Utilisé par le solveur PIR (Phase 4) pour injecter des stimuli optimaux.
 */
export function createSolverPressure(
  targetNodeIds: string[],
  direction:     PressureDirection,
  magnitude:     number,
  metadata:      Record<string, unknown> = {},
): PressureEvent {
  return createPressure({
    targetNodeIds,
    direction,
    magnitude,
    profile:  PressureProfile.impulse,
    source:   PressureSource.solver,
    metadata,
  })
}

/**
 * Génère un event de replay — pour rejouer un log temporel (Phase 5).
 */
export function replayPressureEvent(original: PressureEvent): PressureEvent {
  return {
    ...original,
    id:        nextId('replay'),
    timestamp: Date.now(),
    source:    PressureSource.replay,
    metadata:  { ...original.metadata, replayOf: original.id },
  }
}
