import { useCallback, useEffect, useRef, useState } from 'react'
import {
  fetchFiltered,
  fetchImages,
  fetchSearch,
  type ImageRecord,
  type SearchResponse,
} from './api'
import { remember as rememberResults } from './searchMemory'

export interface Filter {
  person?: string
  event?: string
  q?: string
}

/**
 * The set of photographs the grid is showing.
 *
 * Three sources, and which one applies is decided by the route rather
 * than by a mode flag. With no query and no filter it is the catalog,
 * paged, with a total up front so the grid can pre-allocate its height.
 * With a filter alone it is one call to /search with no query text,
 * which returns the whole filtered pool in catalog order. With a query
 * it is that same endpoint carrying both, because the API applies the
 * filters before scoring - searching within a filter is not the same
 * operation as filtering a search, and only the API can compose them.
 *
 * The two /search cases return `outcome`, which carries the basis line
 * and which images matched. Neither is derivable from the results, so
 * the grid is not the only thing this hook is feeding.
 */
export function useCollection(pageSize: number, filter: Filter) {
  const [images, setImages] = useState<ImageRecord[]>([])
  const [total, setTotal] = useState<number | null>(null)
  const [failed, setFailed] = useState<string | null>(null)
  const [outcome, setOutcome] = useState<SearchResponse | null>(null)
  const [pending, setPending] = useState(false)
  const loading = useRef(false)

  const { person, event, q } = filter
  const searching = Boolean(q)
  const filtered = Boolean(person || event)
  const served = searching || filtered

  useEffect(() => {
    setImages([])
    setTotal(null)
    setFailed(null)
    setOutcome(null)
    loading.current = false

    if (!served) {
      setPending(false)
      return
    }

    let live = true

    setPending(true)

    const request = q
      ? fetchSearch(q, { person, event })
      : fetchFiltered({ person, event })

    request
      .then((found) => {
        if (!live) {
          return
        }

        setImages(found.results)
        setTotal(found.results.length)
        setOutcome(found)

        // The item page is a separate route and cannot re-run a search
        // to fill its diagnostics panel, so what retrieval knew about
        // each result is put somewhere that route can read.
        if (q) {
          rememberResults(q, found)
        }
      })
      .catch((error) => live && setFailed(String(error.message ?? error)))
      .finally(() => {
        if (live) {
          setPending(false)
        }
      })

    return () => {
      live = false
    }
  }, [person, event, q, served])

  const loadMore = useCallback(() => {
    if (served || loading.current) {
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
  }, [served, pageSize, total])

  useEffect(() => {
    if (!served) {
      loadMore()
    }
    // Only when the route changes; later pages are pulled by the grid.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [served, person, event, q])

  const complete = served || (total !== null && images.length >= total)

  return { images, total, failed, complete, outcome, pending, loadMore }
}

export type Collection = ReturnType<typeof useCollection>
