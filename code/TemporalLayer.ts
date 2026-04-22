// ============================================================
// EIDOLON Temporal Layer
// 
//
// RÔLE
// ════
// Enregistrer chaque PressureResult dans le temps.
// Calculer les profils de contrainte par nœud :
//   - Résistance observée sous pression dirigée
//   - Stabilité de belief sur toute la trajectoire
//   - Classification : incompressible / élastique / volatile
//
// SIGNAL CLÉ
// ══════════
// La trajectoire du graphe sous une séquence de pressions est
// le produit analytique final d'EIDOLON.
//
// Un nœud incompressible ne cède pas. Sa belief reste stable
// même sous forte pression. C'est une contrainte structurelle
// que l'entité ne peut pas abandonner sans se dégrader.
//
// Un nœud volatile s'effondre immédiatement sous n'importe
// quelle pression. C'est une dépendance superficielle, non
// constitutive de l'identité de l'entité.
//
// RÉSISTANCE SCORE
// ════════════════
// Pour chaque pression appliquée sur un nœud v :
//   r_i = 1 - |delta_v_i| / magnitude_i
//
// resistanceScore(v) = mean(r_i) sur toutes les pressions
//
// [0] = volatile totalement  [1] = incompressible total
//
// CLASSIFICATION
// ══════════════
// incompressible : resistanceScore > 0.65
// elastic        : 0.35 ≤ resistanceScore ≤ 0.65
// volatile       : resistanceScore < 0.35
// undefined      : < 2 pressions observées
// ============================================================

import type { PressureEvent, PressureResult } from '../types/pressure'
import type { TopologyResult, TopologicalClass } from './BeliefTopology'

export type ConstraintClass = 'incompressible' | 'elastic' | 'volatile' | 'undefined'

export interface TemporalRecord {
  recordId:        string
  timestamp:       number
  event:           PressureEvent
  beliefsBefore:   Map<string, number>
  beliefsAfter:    Map<string, number>
  beliefDeltas:    Map<string, number>
  entropyBefore:   number
  entropyAfter:    number
  entropyDelta:    number
  nodesReached:    number
  totalEnergy:     number
  collapseEvents:  string[]
}

export interface ConstraintProfile {
  nodeId:           string
  label:            string
  category:         string

  // Résistance
  resistanceScore:  number    // [0,1]
  meanDelta:        number    // |delta| moyen observé
  maxDelta:         number    // pic maximum de réponse
  samplesN:         number    // nombre d'observations

  // Trajectoire belief
  beliefMean:       number
  beliefStd:        number
  beliefFirst:      number    // au premier snapshot
  beliefLast:       number    // au dernier snapshot
  trend:            number    // (+) montée / (-) descente
  timesCollapsed:   number

  // Classification
  constraintClass:  ConstraintClass

  // Topologie (Phase 6 — optionnel si TopologyResult disponible)
  topologicalClass:       TopologicalClass | 'unknown'
  topologicalPersistence: number    // persistence de la composante [0,1]
  effectiveResistance:    number    // α·resistanceScore + (1-α)·topologicalPersistence
}

export interface TrajectoryInsight {
  type: 'constraint' | 'volatile' | 'collapse_sequence' | 'entropy' | 'general'
  severity: 'high' | 'medium' | 'low'
  message: string
  nodeIds?: string[]
}

export interface TemporalReport {
  generatedAt:        number
  totalEvents:        number
  totalDuration:      number
  records:            TemporalRecord[]
  constraintProfiles: ConstraintProfile[]
  entropyTrajectory:  Array<{ t: number; entropy: number; label: string }>
  insights:           TrajectoryInsight[]
}

const INCOMPRESSIBLE_THRESHOLD = 0.65
const VOLATILE_THRESHOLD       = 0.35
const MIN_SAMPLES_TO_CLASSIFY  = 2
const MAX_RECORDS              = 200  // plafond mémoire — éviction FIFO au-delà

function clsFromScore(score: number, n: number): ConstraintClass {
  if (n < MIN_SAMPLES_TO_CLASSIFY) return 'undefined'
  if (score > INCOMPRESSIBLE_THRESHOLD) return 'incompressible'
  if (score < VOLATILE_THRESHOLD)       return 'volatile'
  return 'elastic'
}

// ── Classe principale ─────────────────────────────────────────────────────────

export class TemporalLayer {
  private records: TemporalRecord[] = []

  // ── Enregistrement ───────────────────────────────────────────────────────────

  record(event: PressureEvent, result: PressureResult): string {
    const recordId = `trec_${Date.now()}_${this.records.length}`

    this.records.push({
      recordId,
      timestamp:      result.timestamp,
      event,
      beliefsBefore:  new Map(result.beliefsBefore),
      beliefsAfter:   new Map(result.beliefsAfter),
      beliefDeltas:   new Map(result.beliefDeltas),
      entropyBefore:  result.entropyBefore,
      entropyAfter:   result.entropyAfter,
      entropyDelta:   result.entropyAfter - result.entropyBefore,
      nodesReached:   result.wavePropagation.nodesReached,
      totalEnergy:    result.wavePropagation.totalEnergy,
      collapseEvents: [...result.collapseEvents],  // copie — pas de référence partagée
    })

    // Éviction FIFO — maintenir la taille sous MAX_RECORDS
    if (this.records.length > MAX_RECORDS) {
      this.records.shift()
    }

    return recordId
  }

  // ── Historique de beliefs par nœud (pour SemanticAffinity / Pearson) ─────────

  extractBeliefHistory(): Map<string, number[]> {
    const history = new Map<string, number[]>()
    for (const rec of this.records) {
      for (const [nodeId, belief] of rec.beliefsAfter) {
        if (!history.has(nodeId)) history.set(nodeId, [])
        history.get(nodeId)!.push(belief)
      }
    }
    return history
  }

  // ── Accès aux records ────────────────────────────────────────────────────────

  getRecords(): TemporalRecord[] {
    return [...this.records]
  }

  getLength(): number {
    return this.records.length
  }

  clear(): void {
    this.records = []
  }

  // ── Calcul des profils de contrainte ─────────────────────────────────────────

  computeConstraintProfiles(
    canvasNodes: Array<{ id: string; label: string; category: string }>,
    topology?: TopologyResult,
  ): ConstraintProfile[] {
    const profiles: ConstraintProfile[] = []

    for (const node of canvasNodes) {
      const { id, label, category } = node

      // Collect belief values across time
      const beliefSeries: number[] = []
      const deltaAbsSeries: number[] = []
      const resistanceSamples: number[] = []
      const magnitudeSeries: number[] = []   // magnitudes des événements ayant produit un delta
      let timesCollapsed = 0
      let pressuresDirectlyApplied = 0

      for (const rec of this.records) {
        // Belief after this event
        const belief = rec.beliefsAfter.get(id)
        if (belief !== undefined) beliefSeries.push(belief)

        // Absolute delta
        const delta = rec.beliefDeltas.get(id)
        if (delta !== undefined) {
          const absDelta = Math.abs(delta)
          deltaAbsSeries.push(absDelta)
          magnitudeSeries.push(rec.event.pressureVector.magnitude)

          // Check if directly targeted
          const targeted = rec.event.targetNodeIds.includes(id)
          if (targeted) {
            pressuresDirectlyApplied++
            const mag = rec.event.pressureVector.magnitude
            // Resistance = how little it moved relative to magnitude applied
            const r = 1 - Math.min(1, absDelta / Math.max(mag, 0.01))
            resistanceSamples.push(r)
          }
        }

        if (rec.collapseEvents.includes(id)) timesCollapsed++
      }

      if (beliefSeries.length === 0) continue

      const mean = (arr: number[]) => arr.reduce((s, v) => s + v, 0) / arr.length
      const std  = (arr: number[], m: number) =>
        Math.sqrt(arr.reduce((s, v) => s + (v - m) ** 2, 0) / arr.length)

      const beliefMean  = mean(beliefSeries)
      const beliefStd   = std(beliefSeries, beliefMean)
      const meanDelta   = deltaAbsSeries.length > 0 ? mean(deltaAbsSeries) : 0
      const maxDelta    = deltaAbsSeries.length > 0 ? Math.max(...deltaAbsSeries) : 0

      // Resistance score — prefer direct pressure samples; fall back to delta-based estimate
      let resistanceScore: number
      let samplesN: number
      if (resistanceSamples.length >= MIN_SAMPLES_TO_CLASSIFY) {
        resistanceScore = mean(resistanceSamples)
        samplesN = resistanceSamples.length
      } else {
        // Indirect estimate: low delta relative to mean source pressure = resistant
        // Use actual mean pressure magnitude instead of an arbitrary constant
        const meanMag = magnitudeSeries.length > 0 ? mean(magnitudeSeries) : 0.5
        resistanceScore = 1 - Math.min(1, meanDelta / Math.max(meanMag, 0.01))
        samplesN = deltaAbsSeries.length
      }

      // Trend: linear regression slope of belief over time
      let trend = 0
      if (beliefSeries.length >= 2) {
        const n = beliefSeries.length
        const xs = Array.from({ length: n }, (_, i) => i)
        const xMean = (n - 1) / 2
        const slope =
          xs.reduce((s, x, i) => s + (x - xMean) * (beliefSeries[i] - beliefMean), 0) /
          xs.reduce((s, x) => s + (x - xMean) ** 2, 0)
        trend = slope
      }

      // ── Phase 6 : données topologiques ────────────────────────────────────
      const topoProfile = topology?.profiles.get(id)
      const topologicalClass:       TopologicalClass | 'unknown' = topoProfile?.topologicalClass ?? 'unknown'
      const topologicalPersistence: number = topoProfile?.persistence ?? 0
      // α = 0.60 résistance empirique, (1-α) = 0.40 ancrage topologique
      const TOPO_ALPHA      = 0.60
      const effectiveResistance = TOPO_ALPHA * resistanceScore + (1 - TOPO_ALPHA) * topologicalPersistence

      profiles.push({
        nodeId: id,
        label,
        category,
        resistanceScore,
        meanDelta,
        maxDelta,
        samplesN,
        beliefMean,
        beliefStd,
        beliefFirst: beliefSeries[0],
        beliefLast:  beliefSeries[beliefSeries.length - 1],
        trend,
        timesCollapsed,
        constraintClass: clsFromScore(resistanceScore, samplesN),
        topologicalClass,
        topologicalPersistence,
        effectiveResistance,
      })
    }

    // Sort: incompressible first, then by resistanceScore desc
    const CLASS_ORDER: Record<ConstraintClass, number> = {
      incompressible: 0, elastic: 1, volatile: 2, undefined: 3,
    }
    profiles.sort((a, b) =>
      CLASS_ORDER[a.constraintClass] - CLASS_ORDER[b.constraintClass] ||
      b.resistanceScore - a.resistanceScore,
    )

    return profiles
  }

  // ── Insights algorithmiques ───────────────────────────────────────────────────

  generateInsights(profiles: ConstraintProfile[]): TrajectoryInsight[] {
    const insights: TrajectoryInsight[] = []

    const incompressible = profiles.filter(p => p.constraintClass === 'incompressible')
    const volatile_      = profiles.filter(p => p.constraintClass === 'volatile')
    const collapsed      = profiles.filter(p => p.timesCollapsed > 0)

    if (incompressible.length > 0) {
      insights.push({
        type: 'constraint',
        severity: 'high',
        message: `${incompressible.length} contrainte${incompressible.length > 1 ? 's' : ''} incompressible${incompressible.length > 1 ? 's' : ''} identifiée${incompressible.length > 1 ? 's' : ''} : ${incompressible.slice(0, 3).map(p => p.label).join(', ')}${incompressible.length > 3 ? ` +${incompressible.length - 3}` : ''}. Ces nœuds ne cèdent pas sous pression — ils définissent l'architecture cognitive incompressible de la cible.`,
        nodeIds: incompressible.map(p => p.nodeId),
      })
    }

    if (volatile_.length > 0) {
      insights.push({
        type: 'volatile',
        severity: 'medium',
        message: `${volatile_.length} dépendance${volatile_.length > 1 ? 's' : ''} superficielle${volatile_.length > 1 ? 's' : ''} : ${volatile_.slice(0, 3).map(p => p.label).join(', ')}. Ces nœuds cèdent sous pression — ils ne constituent pas un ancrage structurel.`,
        nodeIds: volatile_.map(p => p.nodeId),
      })
    }

    if (collapsed.length > 0) {
      const collapseNames = collapsed.map(p => `${p.label} (×${p.timesCollapsed})`).join(', ')
      insights.push({
        type: 'collapse_sequence',
        severity: 'high',
        message: `Événements de collapse observés : ${collapseNames}. La contrainte a été dépassée — ces nœuds sont au seuil de rupture structurelle.`,
        nodeIds: collapsed.map(p => p.nodeId),
      })
    }

    // Entropy trajectory
    if (this.records.length >= 2) {
      const first = this.records[0]
      const last  = this.records[this.records.length - 1]
      const entropyDrift = last.entropyAfter - first.entropyBefore
      if (Math.abs(entropyDrift) > 0.1) {
        insights.push({
          type: 'entropy',
          severity: 'medium',
          message: entropyDrift < 0
            ? `Entropie en décroissance (${entropyDrift.toFixed(3)} bits) : le graphe converge. Les croyances se cristallisent sous pression accumulée.`
            : `Entropie en hausse (+${entropyDrift.toFixed(3)} bits) : le graphe diverge. La pression génère de l'incertitude — la cible est déstabilisée.`,
        })
      }
    }

    if (this.records.length >= 1 && insights.length === 0) {
      insights.push({
        type: 'general',
        severity: 'low',
        message: 'Données insuffisantes pour identifier des contraintes structurelles. Appliquer davantage de pressions dirigées.',
      })
    }

    return insights
  }

  // ── Rapport complet ───────────────────────────────────────────────────────────

  generateReport(
    canvasNodes: Array<{ id: string; label: string; category: string }>,
    topology?: TopologyResult,
  ): TemporalReport {
    const profiles = this.computeConstraintProfiles(canvasNodes, topology)
    const insights = this.generateInsights(profiles)

    const entropyTrajectory = this.records.map(rec => ({
      t:       rec.timestamp,
      entropy: rec.entropyAfter,
      label:   rec.event.targetNodeIds.slice(0, 2).join('+'),
    }))

    const totalDuration = this.records.length >= 2
      ? this.records[this.records.length - 1].timestamp - this.records[0].timestamp
      : 0

    return {
      generatedAt:        Date.now(),
      totalEvents:        this.records.length,
      totalDuration,
      records:            [...this.records],
      constraintProfiles: profiles,
      entropyTrajectory,
      insights,
    }
  }
}
