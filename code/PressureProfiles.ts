// ============================================================
// EIDOLON Pressure Profiles
// 
//
// Un profil définit la FORME TEMPORELLE d'un stimulus :
// comment la magnitude évolue dans le temps sur N steps.
//
// PROFILS :
//   impulse    — pic instantané au t=0, zéro ensuite
//                Signal analytique : réponse initiale pure
//   ramp       — montée linéaire de 0 à 1 sur la durée
//                Signal analytique : seuil d'activation
//   sustained  — magnitude constante sur toute la durée
//                Signal analytique : résistance à la fatigue
//   oscillating — alternance confirm/contradict (sinus rectifié)
//                Signal analytique : élasticité, hystérésis
//
// Usage :
//   getPressureMultiplier(profile, t, duration)  → [0,1]
//   expandProfile(profile, baseMagnitude, steps) → number[]
// ============================================================

import { PressureProfile } from '../types/pressure'

/**
 * Retourne le multiplicateur de magnitude à l'instant t
 * pour une durée totale `duration`.
 *
 * @param profile  - profil temporel
 * @param t        - instant courant [0, duration]
 * @param duration - durée totale du profil (steps)
 */
export function getPressureMultiplier(
  profile: PressureProfile,
  t: number,
  duration: number,
): number {
  switch (profile) {
    case PressureProfile.impulse:
      // Pic instantané uniquement au premier step
      return t === 0 ? 1.0 : 0.0

    case PressureProfile.ramp:
      // Montée linéaire 0→1
      return duration <= 0 ? 1.0 : Math.min(t / duration, 1.0)

    case PressureProfile.sustained:
      // Pression constante
      return 1.0

    case PressureProfile.oscillating:
      // Sinus rectifié — 4 oscillations sur la durée totale
      if (duration <= 0) return 1.0
      return Math.abs(Math.sin((t / duration) * Math.PI * 4))

    default:
      return 1.0
  }
}

/**
 * Développe un profil en une séquence de magnitudes sur N steps.
 * Le step 0 correspond à t=0, le step N-1 à t=duration.
 *
 * @param profile       - profil temporel
 * @param baseMagnitude - magnitude de base [0, +∞)
 * @param steps         - nombre de steps (défaut: 10)
 * @returns tableau de magnitudes effectives [magnitude × multiplier]
 */
export function expandProfile(
  profile: PressureProfile,
  baseMagnitude: number,
  steps = 10,
): number[] {
  if (steps <= 0) return []
  const duration = steps - 1  // t va de 0 à duration

  const result: number[] = []
  for (let i = 0; i < steps; i++) {
    const multiplier = getPressureMultiplier(profile, i, duration)
    result.push(baseMagnitude * multiplier)
  }
  return result
}

/**
 * Retourne le nombre de steps significatifs d'un profil.
 * impulse → 1 step suffit
 * autres → utilise le défaut N fourni
 */
export function effectiveSteps(profile: PressureProfile, defaultN = 10): number {
  return profile === PressureProfile.impulse ? 1 : defaultN
}
