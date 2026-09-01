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

export async function fetchPalette(): Promise<PaletteResponse> {
  const response = await fetch('/palette')

  if (!response.ok) {
    throw new Error(`/palette responded ${response.status}`)
  }

  return response.json()
}
