import { useCallback, useEffect, useRef, useState } from 'react'
import { fetchFiltered, fetchImages, type ImageRecord } from './api'

export interface Filter {
  person?: string
  event?: string
}

/**
 * The set of photographs the grid is showing.
 *
 * Unfiltered, that is the catalog, paged, with a total up front so the
 * grid can pre-allocate its height. Filtered, it is one call to /search
 * with no query text, which returns the whole filtered pool in catalog
 * order - there is nothing to page towards and no ranking to apply.
 */
export function useCollection(pageSize: number, filter: Filter) {
  const [images, setImages] = useState<ImageRecord[]>([])
  const [total, setTotal] = useState<number | null>(null)
  const [failed, setFailed] = useState<string | null>(null)
  const loading = useRef(false)

  const { person, event } = filter
  const filtered = Boolean(person || event)

  useEffect(() => {
    setImages([])
    setTotal(null)
    setFailed(null)
    loading.current = false

    if (!filtered) {
      return
    }

    let live = true

    fetchFiltered({ person, event })
      .then((found) => {
        if (live) {
          setImages(found.results)
          setTotal(found.results.length)
        }
      })
      .catch((error) => live && setFailed(String(error.message ?? error)))

    return () => {
      live = false
    }
  }, [person, event, filtered])

  const loadMore = useCallback(() => {
    if (filtered || loading.current) {
      return
    }

    setImages((current) => {
      if (total !== null && current.length >= total) {
        return current
      }

      loading.current = true

      fetchImages(current.length, pageSize)
        .then((page) => {
          setTotal(page.total)
          setImages((existing) =>
            existing.length === page.offset
              ? [...existing, ...page.images]
              : existing,
          )
        })
        .catch((error) => setFailed(String(error.message ?? error)))
        .finally(() => {
          loading.current = false
        })

      return current
    })
  }, [filtered, pageSize, total])

  useEffect(() => {
    if (!filtered) {
      loadMore()
    }
    // Only when the filter changes; later pages are pulled by the grid.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filtered, person, event])

  const complete = filtered || (total !== null && images.length >= total)

  return { images, total, failed, complete, loadMore }
}
