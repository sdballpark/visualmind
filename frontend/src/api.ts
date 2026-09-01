/**
 * Types and fetches for the read-only API.
 *
 * Paths are relative. The dev server proxies them to 127.0.0.1, so no
 * port appears in component code and there is no CORS to handle.
 */

export interface PaletteMark {
  sha256: string
  /** Degrees. Null when the photograph has no dominant hue. */
  hue: number | null
  /** 0-1, CIE L*. Null when the palette builder could not read it. */
  lightness: number | null
  /** ISO 8601, or null. 153 of 441 have no capture time. */
  captured: string | null
}

export interface PaletteResponse {
  total: number
  undated: number
  achromatic: number
  marks: PaletteMark[]
}

export interface Dimensions {
  width: number
  height: number
}

export interface ImageRecord {
  sha256: string
  filename: string
  caption: string
  captured: string | null
  grid: Dimensions | null
  lightbox: Dimensions | null
}

export interface ImagesResponse {
  total: number
  offset: number
  limit: number
  images: ImageRecord[]
}

export async function fetchImages(
  offset: number,
  limit: number,
): Promise<ImagesResponse> {
  const response = await fetch(`/images?offset=${offset}&limit=${limit}`)

  if (!response.ok) {
    throw new Error(`/images responded ${response.status}`)
  }

  return response.json()
}

export interface Person {
  name: string
  images: number
  faces: number
}

export interface EventSummary {
  id: string
  name: string
  start: string | null
  end: string | null
  images: number
}

export interface DuplicateMember extends ImageRecord {
  keep: boolean
}

export interface DuplicateGroup {
  group: string
  tier: string
  members: DuplicateMember[]
}

export interface ImageDetail extends ImageRecord {
  /** Position in catalog order, for pulling neighbours from /images. */
  index: number
  total: number
  people: Person[]
  event: EventSummary | null
  duplicates: DuplicateGroup | null
}

export interface SearchResponse {
  results: ImageRecord[]
  basis: string
  score_kind: string
  people: string[]
  events: string[]
  pool_size: number
  corpus_size: number
}

/**
 * The filtered collection.
 *
 * With no query text the API returns the whole filtered pool in catalog
 * order rather than an order manufactured from an empty embedding, so
 * this is the right endpoint for "everything with this person in it".
 */
export async function fetchFiltered(
  filter: { person?: string; event?: string },
): Promise<SearchResponse> {
  const params = new URLSearchParams({ q: '' })

  if (filter.person) {
    params.set('person', filter.person)
  }

  if (filter.event) {
    params.set('event', filter.event)
  }

  const response = await fetch(`/search?${params}`)

  if (!response.ok) {
    const body = await response.json().catch(() => ({}))

    throw new Error(body.detail ?? `/search responded ${response.status}`)
  }

  return response.json()
}

export async function fetchImage(sha256: string): Promise<ImageDetail> {
  const response = await fetch(`/image/${sha256}`)

  if (!response.ok) {
    throw new Error(
      response.status === 404
        ? 'No photograph with that identifier'
        : `/image responded ${response.status}`,
    )
  }

  return response.json()
}

/** Where a lightbox-size image lives. */
export function lightboxImage(sha256: string): string {
  return `/thumbnails/lightbox/${sha256}.jpg`
}

/** Where a grid thumbnail lives. The frontend builds this from the sha. */
export function gridThumbnail(sha256: string): string {
  return `/thumbnails/grid/${sha256}.jpg`
}

export async function fetchPalette(): Promise<PaletteResponse> {
  const response = await fetch('/palette')

  if (!response.ok) {
    throw new Error(`/palette responded ${response.status}`)
  }

  return response.json()
}
