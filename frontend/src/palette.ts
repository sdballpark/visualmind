/**
 * Hue anchors for the density strip.
 *
 * The corpus is bimodal, not evenly spread: of 440 images carrying a
 * hue, 286 fall under 60 degrees and 130 sit between 180 and 240. An
 * evenly spaced palette would spend most of its range on hues almost
 * nothing lands in, and crush the two places everything actually is
 * into one swatch each. So the anchors are placed where the mass is -
 * six across the warm cluster, four across the cool one - with five
 * more spread thinly so the handful of greens and violets are not
 * misreported as something else.
 *
 * Measured distribution, 10-degree buckets:
 *   10-20: 75   20-30: 93   30-40: 60   40-50: 22
 *   200-210: 27   210-220: 62   220-230: 17
 *
 * The rendered colours sit in a narrow lightness band on purpose. Mark
 * height already encodes the photograph's lightness, so an anchor that
 * was much darker or lighter than its neighbours would compete with
 * that reading and make height ambiguous.
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
export const NEUTRAL = '#9c948a'

export const ANCHORS: readonly Anchor[] = [
  // Warm cluster - 286 of 440 images.
  { hue: 355, color: '#a5474e', name: 'rose red' },
  { hue: 8, color: '#ae4f3d', name: 'terracotta' },
  { hue: 18, color: '#b55f3a', name: 'burnt orange' },
  { hue: 26, color: '#b0743e', name: 'amber' },
  { hue: 35, color: '#a58240', name: 'ochre' },
  { hue: 46, color: '#998b45', name: 'olive gold' },

  // Cool cluster - 130 of 440.
  { hue: 195, color: '#4f8a92', name: 'teal' },
  { hue: 206, color: '#4a86a1', name: 'sky' },
  { hue: 215, color: '#4f7fac', name: 'blue' },
  { hue: 227, color: '#5b76ae', name: 'indigo' },

  // Thin cover for everything else - 24 images between them.
  { hue: 75, color: '#84904c', name: 'moss' },
  { hue: 103, color: '#699055', name: 'green' },
  { hue: 160, color: '#4f8f79', name: 'sea green' },
  { hue: 262, color: '#7470ab', name: 'violet' },
  { hue: 315, color: '#9f5f90', name: 'magenta' },
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
