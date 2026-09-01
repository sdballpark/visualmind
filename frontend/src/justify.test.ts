import { describe, expect, it } from 'vitest'
import {
  FALLBACK_RATIO,
  heightOf,
  justify,
  ratioOf,
  visibleRange,
  type Row,
} from './justify'

const OPTIONS = { containerWidth: 1000, targetHeight: 200, gap: 5 }

/** Landscape 3:2, portrait 2:3, square. */
const SHAPES = [1.5, 2 / 3, 1, 1.5, 1.5, 2 / 3, 1.5, 1, 1.5, 1.5, 1.5, 1]

function rowsOf(ratios: number[], options = OPTIONS) {
  return justify(ratios, (r) => r, options)
}

function rowWidth(row: Row<number>) {
  const last = row.items[row.items.length - 1]

  return last.x + last.width
}

/** Enough rows that a buffer difference is resolvable. */
const MANY = Array.from({ length: 200 }, (_, i) => SHAPES[i % SHAPES.length])

describe('ratioOf', () => {
  it('uses the recorded dimensions', () => {
    expect(ratioOf({ width: 400, height: 300 })).toBeCloseTo(4 / 3, 6)
  })

  it('falls back when an image has no thumbnail dimensions', () => {
    // build_thumbnails records these as unreadable rather than dropping
    // them, so the grid has to place something.
    expect(ratioOf(null)).toBe(FALLBACK_RATIO)
    expect(ratioOf({ width: 0, height: 300 })).toBe(FALLBACK_RATIO)
  })
})

describe('justify', () => {
  it('fills the container width exactly on every full row', () => {
    const rows = rowsOf(SHAPES)

    // The last row is left at its natural height, so it is excluded.
    for (const row of rows.slice(0, -1)) {
      expect(rowWidth(row)).toBe(OPTIONS.containerWidth)
    }
  })

  it('keeps every image at its true aspect ratio', () => {
    for (const row of rowsOf(SHAPES)) {
      for (const placed of row.items) {
        // The last item in a row absorbs the rounding remainder, so it
        // is allowed a pixel or two of slack.
        expect(placed.width / placed.height).toBeCloseTo(placed.item, 1)
      }
    }
  })

  it('gives one row one height', () => {
    for (const row of rowsOf(SHAPES)) {
      for (const placed of row.items) {
        expect(placed.height).toBe(row.height)
      }
    }
  })

  it('does not stretch a short last row across the width', () => {
    // One leftover photograph blown up to fill 1000px would tower over
    // every row above it.
    const rows = rowsOf([1.5, 1.5, 1.5, 1.5, 1.5])
    const last = rows[rows.length - 1]

    expect(last.height).toBe(OPTIONS.targetHeight)
    expect(rowWidth(last)).toBeLessThan(OPTIONS.containerWidth)
  })

  it('keeps rows near the target height', () => {
    const heights = rowsOf(SHAPES).slice(0, -1).map((row) => row.height)

    for (const height of heights) {
      expect(Math.abs(height - OPTIONS.targetHeight)).toBeLessThan(60)
    }
  })

  it('stacks rows with a gap and no overlap', () => {
    const rows = rowsOf(SHAPES)

    for (let i = 1; i < rows.length; i += 1) {
      expect(rows[i].top).toBe(
        rows[i - 1].top + rows[i - 1].height + OPTIONS.gap,
      )
    }
  })

  it('places every image exactly once', () => {
    const rows = rowsOf(SHAPES)
    const placed = rows.flatMap((row) => row.items)

    expect(placed).toHaveLength(SHAPES.length)
  })

  it('returns nothing before the container has been measured', () => {
    expect(rowsOf(SHAPES, { ...OPTIONS, containerWidth: 0 })).toEqual([])
    expect(rowsOf([])).toEqual([])
  })
})

describe('heightOf', () => {
  it('is zero with no rows, so the grid can pre-allocate from a total', () => {
    expect(heightOf([], 5)).toBe(0)
  })

  it('reaches past the last row', () => {
    const rows = rowsOf(SHAPES)
    const last = rows[rows.length - 1]

    expect(heightOf(rows, OPTIONS.gap)).toBe(last.top + last.height + 5)
  })
})

describe('visibleRange', () => {
  const rows = rowsOf(MANY)

  it('mounts far fewer rows than exist', () => {
    const [first, last] = visibleRange(rows, 0, 100, 1, 1.4)

    expect(last - first + 1).toBeLessThan(rows.length)
  })

  it('weights the buffer towards where the reader is going', () => {
    const middle = rows[Math.floor(rows.length / 2)].top

    // The viewport has to be large enough that the difference between
    // the two buffers spans a row; at 100px it cannot, and the test
    // passes on a grid that buffers symmetrically.
    const down = visibleRange(rows, middle, 600, 1, 1.4)
    const up = visibleRange(rows, middle, 600, -1, 1.4)

    // Scrolling down reaches further ahead than scrolling up does.
    expect(down[1]).toBeGreaterThan(up[1])
    // Scrolling up keeps more of what is behind.
    expect(up[0]).toBeLessThan(down[0])
  })

  it('reports an empty range rather than a bogus one when nothing fits', () => {
    expect(visibleRange([], 0, 100, 1, 1.4)).toEqual([0, -1])
  })
})
