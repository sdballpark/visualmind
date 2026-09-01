import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { gridThumbnail, type ImageRecord } from './api'
import { heightOf, justify, ratioOf, visibleRange } from './justify'
import { currentPath, navigate, recall } from './router'
import type { Collection } from './useCollection'

/**
 * The collection as a justified grid.
 *
 * Layout is computed from the thumbnail dimensions in the payload, so
 * rows are correct before any image loads and stay put as they arrive.
 * Only the rows near the viewport are mounted, with the buffer weighted
 * towards wherever the reader is going. 441 images do not need that;
 * fitting it into a working grid later would mean rewriting it.
 */

const TARGET_HEIGHT = 208
const GAP = 5
const BUFFER = 1.4

/** Start another page once this close to the end of what is laid out. */
const PREFETCH_ROWS = 4

function useViewport() {
  const [state, setState] = useState({ scrollTop: 0, height: 0 })
  const direction = useRef<1 | -1>(1)
  const previous = useRef(0)

  useEffect(() => {
    const read = () => {
      const scrollTop = window.scrollY

      direction.current = scrollTop >= previous.current ? 1 : -1
      previous.current = scrollTop

      setState({ scrollTop, height: window.innerHeight })
    }

    read()
    window.addEventListener('scroll', read, { passive: true })
    window.addEventListener('resize', read)

    return () => {
      window.removeEventListener('scroll', read)
      window.removeEventListener('resize', read)
    }
  }, [])

  return { ...state, direction: direction.current }
}

/*
 * The collection is a prop rather than a hook call here. The basis line
 * and the strip's highlighting need the same outcome this grid is
 * drawn from, and fetching it twice would let the sentence on the page
 * describe a different search than the images under it.
 */
export function PhotoGrid({ collection }: { collection: Collection }) {
  const { images, total, failed, complete, loadMore } = collection
  const container = useRef<HTMLDivElement | null>(null)
  const [width, setWidth] = useState(0)
  const [offsetTop, setOffsetTop] = useState(0)
  const viewport = useViewport()

  useLayoutEffect(() => {
    const element = container.current

    if (!element) {
      return
    }

    const measure = () => {
      setWidth(element.getBoundingClientRect().width)
      setOffsetTop(element.offsetTop)
    }

    measure()

    const observer = new ResizeObserver(measure)

    observer.observe(element)
    window.addEventListener('resize', measure)

    return () => {
      observer.disconnect()
      window.removeEventListener('resize', measure)
    }
  }, [images.length])

  const rows = useMemo(
    () =>
      justify<ImageRecord>(images, (image) => ratioOf(image.grid), {
        containerWidth: width,
        targetHeight: TARGET_HEIGHT,
        gap: GAP,
      }),
    [images, width],
  )

  const laidOut = heightOf(rows, GAP)

  /*
   * Height for what has not loaded yet, so the scrollbar is
   * representative from the first page rather than growing as later
   * pages land. The per-image figure comes from the rows already laid
   * out, so it reflects this collection's shapes rather than a guess.
   */
  const perImage = images.length ? laidOut / images.length : 0
  const remaining = total === null ? 0 : Math.max(0, total - images.length)
  const height = laidOut + remaining * perImage

  const [first, last] = visibleRange(
    rows,
    Math.max(0, viewport.scrollTop - offsetTop),
    viewport.height,
    viewport.direction,
    BUFFER,
  )

  /*
   * Replay the scroll position the reader left from, once the grid is
   * tall enough to hold it. Restoring earlier would clamp against a
   * document that has not been allocated yet and land them at the top.
   */
  const restored = useRef(false)

  useEffect(() => {
    if (restored.current) {
      return
    }

    const wanted = recall(currentPath())

    if (wanted > 0 && height >= wanted) {
      window.scrollTo(0, wanted)
      restored.current = true
    }
  }, [height])

  useEffect(() => {
    if (complete || rows.length === 0) {
      return
    }

    if (last >= rows.length - 1 - PREFETCH_ROWS) {
      loadMore()
    }
  }, [last, rows.length, complete, loadMore])

  return (
    <div className="grid" ref={container} style={{ height }}>
      {failed && <p className="failed">{failed}</p>}

      {rows.slice(first, last + 1).map((row, index) => (
        <div
          className="row"
          key={first + index}
          style={{ top: row.top, height: row.height }}
        >
          {row.items.map((placed) => (
            <a
              key={placed.item.sha256}
              className="photo"
              href={`/photo/${placed.item.sha256}`}
              style={{
                left: placed.x,
                width: placed.width,
                height: placed.height,
              }}
              onClick={(event) => {
                if (event.metaKey || event.ctrlKey || event.shiftKey) {
                  return
                }

                event.preventDefault()
                navigate(`/photo/${placed.item.sha256}`)
              }}
            >
              <img
                src={gridThumbnail(placed.item.sha256)}
                alt={placed.item.caption || placed.item.filename}
                width={placed.width}
                height={placed.height}
                loading="lazy"
                decoding="async"
              />
            </a>
          ))}
        </div>
      ))}
    </div>
  )
}
