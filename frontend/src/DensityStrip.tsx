import { useLayoutEffect, useMemo, useRef, useState } from 'react'
import type { PaletteMark } from './api'
import { colorFor } from './palette'

/**
 * Every photograph in the collection, as one band.
 *
 * Position is capture time, colour is hue, height is lightness. Marks
 * overlap where photographs cluster in time, and that overlap is the
 * density the band is named for - it is not a defect to be spaced out.
 *
 * The undated images are not hidden. They hold their own segment,
 * sized from the count the API reports rather than a constant, because
 * that count has already moved once.
 */

export const BAND = 84
const MIN_MARK = 5
const MAX_MARK = 72
const MARK_WIDTH = 3
const SEGMENT_GAP = 40

/** Lightness for a mark whose palette row carried none. */
const ASSUMED_LIGHTNESS = 0.5

interface Placed {
  key: string
  x: number
  height: number
  color: string
}

function markHeight(lightness: number | null): number {
  const value = lightness ?? ASSUMED_LIGHTNESS

  return MIN_MARK + value * (MAX_MARK - MIN_MARK)
}

/**
 * The container's width, measured before the browser paints.
 *
 * Measured synchronously rather than waiting for the observer's first
 * callback. A band that renders nothing until an async callback arrives
 * is one timing quirk away from an empty page, and it was exactly that
 * in a headless browser.
 */
function useWidth(): [React.RefObject<HTMLDivElement | null>, number] {
  const ref = useRef<HTMLDivElement | null>(null)
  const [width, setWidth] = useState(0)

  useLayoutEffect(() => {
    const element = ref.current

    if (!element) {
      return
    }

    setWidth(element.getBoundingClientRect().width)

    const observer = new ResizeObserver((entries) => {
      setWidth(entries[0].contentRect.width)
    })

    observer.observe(element)

    return () => observer.disconnect()
  }, [])

  return [ref, width]
}

export function place(marks: PaletteMark[], width: number) {
  const dated = marks
    .filter((mark) => mark.captured !== null)
    .sort((a, b) => a.captured!.localeCompare(b.captured!))
  const undated = marks.filter((mark) => mark.captured === null)

  // The undated segment is sized from how many there actually are.
  const share = marks.length ? undated.length / marks.length : 0
  const usable = Math.max(0, width - SEGMENT_GAP - MARK_WIDTH)
  const undatedWidth = usable * share
  const datedWidth = usable - undatedWidth
  const undatedLeft = datedWidth + SEGMENT_GAP

  const times = dated.map((mark) => Date.parse(mark.captured!))
  const earliest = times.length ? Math.min(...times) : 0
  const latest = times.length ? Math.max(...times) : 1
  const span = Math.max(1, latest - earliest)

  const placed: Placed[] = dated.map((mark, index) => ({
    key: mark.sha256,
    x: ((times[index] - earliest) / span) * datedWidth,
    height: markHeight(mark.lightness),
    color: colorFor(mark.hue),
  }))

  const step = undated.length > 1 ? undatedWidth / (undated.length - 1) : 0

  undated.forEach((mark, index) => {
    placed.push({
      key: mark.sha256,
      x: undatedLeft + index * step,
      height: markHeight(mark.lightness),
      color: colorFor(mark.hue),
    })
  })

  return {
    placed,
    datedWidth,
    undatedLeft,
    undatedWidth,
    earliest: times.length ? new Date(earliest) : null,
    latest: times.length ? new Date(latest) : null,
    datedCount: dated.length,
    undatedCount: undated.length,
  }
}

function year(date: Date | null): string {
  return date ? String(date.getUTCFullYear()) : ''
}

export function DensityStrip({ marks }: { marks: PaletteMark[] }) {
  const [ref, width] = useWidth()

  const layout = useMemo(() => place(marks, width), [marks, width])

  return (
    <div className="strip" ref={ref}>
      {width > 0 && (
        <>
          <svg
            className="band"
            width={width}
            height={BAND}
            role="img"
            aria-label={
              `${marks.length} photographs, ${layout.datedCount} placed by ` +
              `capture time and ${layout.undatedCount} undated`
            }
          >
            {layout.placed.map((mark, index) => (
              <rect
                key={mark.key}
                x={mark.x}
                y={(BAND - mark.height) / 2}
                width={MARK_WIDTH}
                height={mark.height}
                fill={mark.color}
                style={{ animationDelay: `${Math.min(index, 440) * 0.8}ms` }}
              />
            ))}
          </svg>

          <div className="legend" style={{ height: 18 }}>
            <span style={{ left: 0 }}>{year(layout.earliest)}</span>
            <span
              style={{ left: layout.datedWidth, transform: 'translateX(-100%)' }}
            >
              {year(layout.latest)}
            </span>
            <span style={{ left: layout.undatedLeft }}>
              undated · {layout.undatedCount}
            </span>
          </div>
        </>
      )}
    </div>
  )
}
