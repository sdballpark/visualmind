import { describe, expect, it } from 'vitest'
import { place } from './DensityStrip'
import {
  ANCHORS,
  NEUTRAL,
  NEUTRAL_MUTED,
  colorFor,
  hueDistance,
  mutedColorFor,
} from './palette'
import type { PaletteMark } from './api'

const WIDTH = 1000

function mark(over: Partial<PaletteMark>): PaletteMark {
  return {
    sha256: Math.random().toString(36).slice(2),
    hue: 20,
    lightness: 0.5,
    captured: '2020-01-01T00:00:00',
    ...over,
  }
}

describe('palette anchors', () => {
  it('gives every anchor a muted twin', () => {
    for (const anchor of ANCHORS) {
      expect(anchor.muted).toMatch(/^#[0-9a-f]{6}$/)
      expect(anchor.muted).not.toBe(anchor.color)
    }
  })

  it('sends a null hue to the neutral, not to an anchor', () => {
    expect(colorFor(null)).toBe(NEUTRAL)
  })

  it('concentrates anchors where the corpus actually is', () => {
    const warm = ANCHORS.filter((a) => a.hue < 60 || a.hue > 340)
    const cool = ANCHORS.filter((a) => a.hue >= 180 && a.hue <= 240)

    // 286 of 440 images are warm, 130 cool, 24 everywhere else.
    expect(warm.length).toBe(4)
    expect(cool.length).toBe(3)
    expect(ANCHORS.length).toBe(8)
    expect(warm.length + cool.length).toBeGreaterThan(ANCHORS.length / 2)
  })

  it('separates the two densest buckets rather than merging them', () => {
    // 20-30 degrees holds 93 images and 210-220 holds 62.
    expect(colorFor(25)).not.toBe(colorFor(215))
  })

  it('gives the dense warm bucket more than one colour', () => {
    // An evenly spaced palette would flatten 10-50 into one swatch.
    const shades = new Set([10, 20, 30, 45].map(colorFor))

    expect(shades.size).toBe(4)
  })

  it('measures hue distance the short way round the wheel', () => {
    expect(hueDistance(355, 5)).toBe(10)
    expect(colorFor(358)).toBe(colorFor(356))
  })
})

describe('placement', () => {
  it('places every mark, dated or not', () => {
    const marks = [
      mark({ captured: '2004-01-01T00:00:00' }),
      mark({ captured: null }),
      mark({ captured: '2024-01-01T00:00:00' }),
    ]

    expect(place(marks, WIDTH).placed).toHaveLength(3)
  })

  it('sizes the undated segment from the count, not a constant', () => {
    const marks = [
      ...Array.from({ length: 3 }, () => mark({})),
      ...Array.from({ length: 1 }, () => mark({ captured: null })),
    ]

    const layout = place(marks, WIDTH)
    const usable = layout.datedWidth + layout.undatedWidth

    expect(layout.undatedWidth / usable).toBeCloseTo(0.25, 5)
  })

  it('follows the count when it moves', () => {
    // 153 was 118 in an earlier spec; the layout must not assume either.
    const many = [
      ...Array.from({ length: 1 }, () => mark({})),
      ...Array.from({ length: 3 }, () => mark({ captured: null })),
    ]

    const layout = place(many, WIDTH)
    const usable = layout.datedWidth + layout.undatedWidth

    expect(layout.undatedWidth / usable).toBeCloseTo(0.75, 5)
  })

  it('orders dated marks by time and spans the dated width', () => {
    const marks = [
      mark({ captured: '2026-01-01T00:00:00' }),
      mark({ captured: '2000-01-01T00:00:00' }),
      mark({ captured: '2013-01-01T00:00:00' }),
    ]

    const layout = place(marks, WIDTH)
    const xs = layout.placed.slice(0, 3).map((p) => p.x)

    expect(xs[0]).toBe(0)
    expect(xs[2]).toBeCloseTo(layout.datedWidth, 5)
    expect(xs[1]).toBeGreaterThan(xs[0])
    expect(xs[1]).toBeLessThan(xs[2])
  })

  it('keeps the undated segment clear of the dated one', () => {
    const marks = [mark({}), mark({ captured: null })]
    const layout = place(marks, WIDTH)
    const undated = layout.placed[layout.placed.length - 1]

    expect(undated.x).toBeGreaterThan(layout.datedWidth)
  })

  it('spaces undated marks evenly, having no time to place them by', () => {
    const marks = [
      mark({}),
      ...Array.from({ length: 3 }, () => mark({ captured: null })),
    ]

    const [, a, b, c] = place(marks, WIDTH).placed

    expect(b.x - a.x).toBeCloseTo(c.x - b.x, 5)
  })

  it('turns lightness into height and leaves hue out of it', () => {
    const dark = place([mark({ lightness: 0.16 })], WIDTH).placed[0]
    const bright = place([mark({ lightness: 0.95 })], WIDTH).placed[0]

    expect(bright.height).toBeGreaterThan(dark.height * 2)
    expect(dark.color).toBe(bright.color)
  })

  it('gives a hueless mark the neutral and still a height', () => {
    const [only] = place([mark({ hue: null, lightness: 0.7 })], WIDTH).placed

    expect(only.color).toBe(NEUTRAL)
    expect(only.height).toBeGreaterThan(0)
  })

  it('survives a collection with no dated images at all', () => {
    const marks = Array.from({ length: 4 }, () => mark({ captured: null }))
    const layout = place(marks, WIDTH)

    expect(layout.placed).toHaveLength(4)
    expect(layout.earliest).toBeNull()
    expect(layout.datedCount).toBe(0)
  })

  it('survives a zero width before the first measurement', () => {
    expect(() => place([mark({})], 0)).not.toThrow()
  })

  it('draws undated marks muted and dated ones at full chroma', () => {
    // 153 undated images at full chroma read as the headline of the
    // band rather than as a footnote to it.
    const marks = [mark({ hue: 25 }), mark({ hue: 25, captured: null })]
    const [dated, undated] = place(marks, WIDTH).placed

    expect(dated.color).toBe(colorFor(25))
    expect(undated.color).toBe(mutedColorFor(25))
    expect(undated.color).not.toBe(dated.color)
  })

  it('draws undated marks narrower, so the segment is a field', () => {
    // The width share stays proportional to the count; only the mark
    // narrows, which opens a gap where there was none.
    const marks = [mark({}), mark({ captured: null })]
    const [dated, undated] = place(marks, WIDTH).placed

    expect(undated.width).toBeLessThan(dated.width)
  })

  it('mutes an achromatic undated mark too', () => {
    const marks = [mark({ hue: null, captured: null })]
    const [only] = place(marks, WIDTH).placed

    expect(only.color).toBe(NEUTRAL_MUTED)
    expect(only.color).not.toBe(NEUTRAL)
  })
})

describe('highlighting a result set', () => {
  const marks = [
    mark({ sha256: 'in-1', captured: '2020-01-01T00:00:00' }),
    mark({ sha256: 'out-1', captured: '2021-01-01T00:00:00' }),
    mark({ sha256: 'in-2', captured: null }),
    mark({ sha256: 'out-2', captured: null }),
  ]

  it('lights every mark when nothing is highlighted', () => {
    // The resting strip has no recessive marks, rather than a set of
    // them that happens to be all of them.
    const layout = place(marks, WIDTH)

    expect(layout.placed.every((placed) => placed.lit)).toBe(true)
    expect(layout.litCount).toBe(marks.length)
  })

  it('still places every mark when a search is on', () => {
    // A search changes what is emphasised, not what exists. Dropping
    // the rest would turn "this is the whole collection" into "this is
    // the whole collection, sometimes".
    const layout = place(marks, WIDTH, new Set(['in-1', 'in-2']))

    expect(layout.placed).toHaveLength(marks.length)
    expect(layout.datedCount).toBe(2)
    expect(layout.undatedCount).toBe(2)
  })

  it('lights only the results, dated and undated alike', () => {
    const layout = place(marks, WIDTH, new Set(['in-1', 'in-2']))
    const lit = layout.placed.filter((placed) => placed.lit).map((p) => p.key)

    expect(lit.sort()).toEqual(['in-1', 'in-2'])
    expect(layout.litCount).toBe(2)
  })

  it('lights nothing when a search returned nothing', () => {
    const layout = place(marks, WIDTH, new Set<string>())

    expect(layout.litCount).toBe(0)
    expect(layout.placed).toHaveLength(marks.length)
  })
})
