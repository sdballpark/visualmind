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
