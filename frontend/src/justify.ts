/**
 * Justified rows: every row fills the container, nothing is cropped.
 *
 * Row heights come from aspect ratios alone, so the whole layout is
 * known before a single image byte arrives. That is the point - a grid
 * that sizes itself from loaded images reflows as they arrive, and
 * reflowing under the reader is the one thing a photo grid must not do.
 */

export interface Placement<T> {
  item: T
  width: number
  height: number
  /** Offset from the row's left edge. */
  x: number
}

export interface Row<T> {
  items: Placement<T>[]
  height: number
  /** Offset from the top of the grid. */
  top: number
}

export interface JustifyOptions {
  containerWidth: number
  targetHeight: number
  gap: number
}

/** Fallback shape for an image with no recorded thumbnail dimensions. */
export const FALLBACK_RATIO = 3 / 2

export function ratioOf(size: { width: number; height: number } | null) {
  if (!size || size.width <= 0 || size.height <= 0) {
    return FALLBACK_RATIO
  }

  return size.width / size.height
}

/**
 * Lay a row out, optionally flush to the container width.
 *
 * Widths are rounded to whole pixels and the remainder is given to the
 * last item, so full rows end flush instead of a pixel short or long.
 * Half a pixel of error per image is invisible; a ragged right edge is
 * not.
 *
 * `flush` is false for the final row, which is deliberately left at its
 * natural width. Applying the remainder there would hand the entire
 * unused width to the last photograph and stretch it - the exact thing
 * not stretching the last row is meant to avoid.
 */
function layRow<T>(
  items: T[],
  ratios: number[],
  height: number,
  options: JustifyOptions,
  top: number,
  flush: boolean,
): Row<T> {
  const { gap } = options

  let x = 0
  const placements: Placement<T>[] = items.map((item, index) => {
    const width = Math.round(ratios[index] * height)
    const placement = { item, width, height, x }

    x += width + gap

    return placement
  })

  const last = placements[placements.length - 1]
  const overflow = x - gap - options.containerWidth

  if (flush && last && overflow !== 0) {
    last.width -= overflow
  }

  return { items: placements, height, top }
}

export function justify<T>(
  items: T[],
  ratioFor: (item: T) => number,
  options: JustifyOptions,
): Row<T>[] {
  const { containerWidth, targetHeight, gap } = options

  if (containerWidth <= 0 || items.length === 0) {
    return []
  }

  const rows: Row<T>[] = []

  let pending: T[] = []
  let ratios: number[] = []
  let ratioSum = 0
  let top = 0

  const flush = (height: number, toWidth = true) => {
    const row = layRow(pending, ratios, height, options, top, toWidth)

    rows.push(row)
    top += height + gap
    pending = []
    ratios = []
    ratioSum = 0
  }

  for (const item of items) {
    const ratio = ratioFor(item)

    pending.push(item)
    ratios.push(ratio)
    ratioSum += ratio

    const gaps = gap * (pending.length - 1)
    const naturalWidth = ratioSum * targetHeight + gaps

    if (naturalWidth >= containerWidth) {
      const withItem = (containerWidth - gaps) / ratioSum

      /*
       * Taking the overflowing image always is what makes rows uneven:
       * one wide photograph can drag a row well below the target while
       * the row before it would have been closer without. Compare the
       * two heights and keep whichever lands nearer the target, which
       * evens the rhythm without changing anything about the fit.
       */
      const withoutGaps = gap * (pending.length - 2)
      const withoutItem =
        pending.length > 1
          ? (containerWidth - withoutGaps) / (ratioSum - ratio)
          : withItem

      if (
        pending.length > 1 &&
        Math.abs(withoutItem - targetHeight) <
          Math.abs(withItem - targetHeight)
      ) {
        const held = pending.pop() as T

        ratios.pop()
        ratioSum -= ratio

        flush(withoutItem)

        pending.push(held)
        ratios.push(ratio)
        ratioSum = ratio
      } else {
        flush(withItem)
      }
    }
  }

  if (pending.length > 0) {
    // The last row keeps its natural height and its natural width.
    // Stretching either would blow a single leftover photograph up to
    // the size of a whole row.
    flush(targetHeight, false)
  }

  return rows
}

/** Total laid-out height, including the trailing gap-free edge. */
export function heightOf<T>(rows: Row<T>[], gap: number): number {
  if (rows.length === 0) {
    return 0
  }

  const last = rows[rows.length - 1]

  return last.top + last.height + gap
}

/**
 * Which rows to render.
 *
 * The buffer is deliberately lopsided: most of it goes where the reader
 * is heading, because that is where the next images are needed, and a
 * symmetric buffer spends half its work behind them.
 */
export function visibleRange<T>(
  rows: Row<T>[],
  scrollTop: number,
  viewport: number,
  direction: 1 | -1,
  buffer: number,
): [number, number] {
  const ahead = viewport * buffer
  const behind = viewport * buffer * 0.25

  const top = scrollTop - (direction === 1 ? behind : ahead)
  const bottom = scrollTop + viewport + (direction === 1 ? ahead : behind)

  let first = rows.length
  let last = -1

  rows.forEach((row, index) => {
    if (row.top + row.height >= top && row.top <= bottom) {
      first = Math.min(first, index)
      last = Math.max(last, index)
    }
  })

  return last < 0 ? [0, -1] : [first, last]
}
