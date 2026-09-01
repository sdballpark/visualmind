import { useEffect, useState } from 'react'
import { fetchPalette, type PaletteResponse } from './api'
import { DensityStrip } from './DensityStrip'
import { PhotoGrid } from './PhotoGrid'

export default function App() {
  const [palette, setPalette] = useState<PaletteResponse | null>(null)
  const [failed, setFailed] = useState<string | null>(null)

  useEffect(() => {
    fetchPalette().then(setPalette).catch((error) => {
      setFailed(String(error.message ?? error))
    })
  }, [])

  return (
    <main>
      <header>
        <h1>VisualMind</h1>
        {palette && (
          <p>
            {palette.total} photographs — the whole collection
          </p>
        )}
      </header>

      {failed && <p className="failed">{failed}</p>}
      {palette && <DensityStrip marks={palette.marks} />}

      <PhotoGrid />
    </main>
  )
}
