import { useEffect, useState } from 'react'
import { fetchPalette, type PaletteResponse } from './api'
import { DensityStrip } from './DensityStrip'
import { PhotoGrid } from './PhotoGrid'
import { PhotoPage } from './PhotoPage'
import { navigate, useRoute } from './router'

function Collection({ person, event }: { person?: string; event?: string }) {
  const [palette, setPalette] = useState<PaletteResponse | null>(null)
  const [failed, setFailed] = useState<string | null>(null)
  const filtered = Boolean(person || event)

  useEffect(() => {
    fetchPalette().then(setPalette).catch((error) => {
      setFailed(String(error.message ?? error))
    })
  }, [])

  return (
    <>
      <header>
        <h1>VisualMind</h1>
        {filtered ? (
          <p>
            <span className="filter">{person ?? event}</span>
            <a
              href="/"
              onClick={(clicked) => {
                clicked.preventDefault()
                navigate('/')
              }}
            >
              all {palette ? palette.total : ''} photographs
            </a>
          </p>
        ) : (
          palette && (
            <p>
              {palette.total} photographs — the whole collection
            </p>
          )
        )}
      </header>

      {failed && <p className="failed">{failed}</p>}
      {/* The strip always shows the whole collection: it is the claim
          that this is everything, which a filter does not change. */}
      {palette && <DensityStrip marks={palette.marks} />}

      <PhotoGrid filter={{ person, event }} />
    </>
  )
}

export default function App() {
  const route = useRoute()

  return (
    <main>
      {route.name === 'photo' ? (
        <PhotoPage sha={route.sha} />
      ) : (
        <Collection person={route.person} event={route.event} />
      )}
    </main>
  )
}
