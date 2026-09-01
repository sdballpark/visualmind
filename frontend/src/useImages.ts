import { useCallback, useEffect, useRef, useState } from 'react'
import { fetchImages, type ImageRecord } from './api'

/**
 * Pages /images, keeping the total so the grid can pre-allocate height.
 *
 * The total arrives with the first page, which is what lets the scroll
 * height be right from the first frame instead of growing under the
 * reader as pages land.
 */
export function useImages(pageSize: number) {
  const [images, setImages] = useState<ImageRecord[]>([])
  const [total, setTotal] = useState<number | null>(null)
  const [failed, setFailed] = useState<string | null>(null)
  const loading = useRef(false)

  const loadMore = useCallback(() => {
    if (loading.current) {
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
  }, [pageSize, total])

  useEffect(() => {
    loadMore()
    // Only on mount: later pages are pulled by the grid as it scrolls.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const complete = total !== null && images.length >= total

  return { images, total, failed, complete, loadMore }
}
