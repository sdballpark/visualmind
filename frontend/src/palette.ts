/**
 * Hue anchors for the density strip.
 *
 * The corpus is not a spectrum. Of 440 images carrying a hue, 286 sit
 * under 60 degrees and 130 between 180 and 240 - a hard warm/cool
 * split with about 24 images scattered everywhere else.
 *
 * Measured distribution, 10-degree buckets:
 *   10-20: 75   20-30: 93   30-40: 60   40-50: 22
 *   200-210: 27   210-220: 62   220-230: 17
 *
 * An earlier version spread 15 anchors over that. It rendered as
 * confetti: enough separate colours that the eye read them as unrelated
 * rather than as two families, which competed with the density the band
 * exists to show. Eight is enough to carry variation inside each family
 * without the families dissolving - four across the warm cluster, three
 * across the cool one, and one absorbing the greens. The remaining
 * stragglers fall to whichever family is nearest rather than each
 * earning a colour of their own.
 *
 * Every anchor sits at L* 52, within 0.21 of the others. Mark height
 * already encodes the photograph's lightness, so an anchor lighter or
 * darker than its neighbours would read as a height difference that is
 * not there. Hue and chroma carry the family; lightness carries nothing.
 *
 * This file is data. Retuning the palette should never require touching
 * the strip's layout.
 */

export interface Anchor {
  /** Where this anchor sits on the wheel, in degrees. */
  readonly hue: number
  /** What it renders as. */
  readonly color: string
  readonly name: string
}

/** Images with no dominant hue: grey, black or white throughout. */
export const NEUTRAL = '#9a9086'

export const ANCHORS: readonly Anchor[] = [
  // Warm family - 285 of 440 images land here.
  { hue: 6, color: '#c35d52', name: 'terracotta' },
  { hue: 18, color: '#b66745', name: 'burnt orange' },
  { hue: 28, color: '#a57142', name: 'amber' },
  { hue: 40, color: '#937842', name: 'ochre' },

  // The greens, and the nearest thing the strip has to a third family
  // at 20 images. Kept muted so it reads as an edge of the warm side
  // rather than as a family of its own.
  { hue: 90, color: '#68854a', name: 'moss' },

  // Cool family - 135 of 440.
  { hue: 200, color: '#51829b', name: 'teal' },
  { hue: 212, color: '#517fb3', name: 'blue' },
  { hue: 224, color: '#627abc', name: 'indigo' },
]

/** Shortest distance between two angles, in degrees. */
export function hueDistance(first: number, second: number): number {
  return Math.abs(((first - second + 180) % 360 + 360) % 360 - 180)
}

/**
 * The colour for a mark.
 *
 * A null hue is an achromatic photograph, which is a real answer rather
 * than a missing one, so it renders neutral instead of disappearing.
 */
export function colorFor(hue: number | null): string {
  if (hue === null || Number.isNaN(hue)) {
    return NEUTRAL
  }

  let closest = ANCHORS[0]
  let smallest = hueDistance(hue, closest.hue)

  for (const anchor of ANCHORS) {
    const distance = hueDistance(hue, anchor.hue)

    if (distance < smallest) {
      smallest = distance
      closest = anchor
    }
  }

  return closest.color
}
