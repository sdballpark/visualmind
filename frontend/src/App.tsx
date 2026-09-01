import { useEffect, useMemo, useState } from 'react'
import { fetchPalette, type PaletteResponse } from './api'
import { DensityStrip } from './DensityStrip'
import { PhotoGrid } from './PhotoGrid'
import { PhotoPage } from './PhotoPage'
import { navigate, pathFor, useRoute } from './router'
import { useCollection } from './useCollection'

const PAGE = 120

/**
 * The query field.
 *
 * A line, not a box. There is no border, no rounded rectangle and no
 * magnifying glass, because the page has none of those anywhere else -
 * a control that announces itself as a Search Widget would be the only
 * chrome on a page whose whole register is that there is none.
 *
 * Submitting navigates rather than fetching directly, so a query is a
 * URL: shareable, reloadable, and reachable with the back button. The
 * fetch follows from the route, which keeps one path into the API.
 */
function SearchField({
  query,
  person,
  event,
  busy,
}: {
  query: string
  person?: string
  event?: string
  busy: boolean
}) {
  const [text, setText] = useState(query)

  return (
    <form
      className="search"
      onSubmit={(submitted) => {
        submitted.preventDefault()
        navigate(
          pathFor({ name: 'grid', person, event, q: text.trim() || undefined }),
        )
      }}
    >
      <input
        type="text"
        value={text}
        onChange={(changed) => setText(changed.target.value)}
        placeholder="Search the collection"
        aria-label="Search the collection"
        autoComplete="off"
        spellCheck={false}
      />
      {busy && <span className="working">searching</span>}
    </form>
  )
}

function Collection({
  person,
  event,
  q,
}: {
  person?: string
  event?: string
  q?: string
}) {
  const [palette, setPalette] = useState<PaletteResponse | null>(null)
  const [failed, setFailed] = useState<string | null>(null)
  const filtered = Boolean(person || event)

  const collection = useCollection(PAGE, { person, event, q })
  const { outcome, pending } = collection

  useEffect(() => {
    fetchPalette().then(setPalette).catch((error) => {
      setFailed(String(error.message ?? error))
    })
  }, [])

  /*
   * Which marks the strip keeps at full weight. Only a text query
   * highlights: a filter alone already narrows the grid, and lighting
   * up its pool would say the filter is a result set rather than the
   * collection it is drawn from.
   */
  const highlight = useMemo(() => {
    if (!q || !outcome) {
      return undefined
    }

    return new Set(outcome.results.map((result) => result.sha256))
  }, [q, outcome])

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
          palette &&
          !q && (
            <p>
              {palette.total} photographs — the whole collection
            </p>
          )
        )}
      </header>

      {/*
        * Keyed by the route's query so the field is remounted when the
        * route changes rather than synchronised to it by an effect. The
        * back button then moves the text along with the results, instead
        * of leaving a stale query sitting above the collection.
        */}
      <SearchField
        key={q ?? ''}
        query={q ?? ''}
        person={person}
        event={event}
        busy={pending}
      />

      {/*
        * The system explaining what it did, in its own words. retrieval
        * composes this sentence; rendering it here rather than writing
        * a second one is what keeps the page from being able to
        * disagree with the search that produced it.
        */}
      {q && outcome && (
        <div className="basis">
          {/*
            * The count sits in the quiet register, above the sentence,
            * rather than being spliced into it. retrieval composes the
            * sentence and deliberately leaves the count out of it,
            * because all three consumers print their own; splicing one
            * back in here would put it on the page twice.
            *
            * The denominator is the pool that was searched, not the
            * corpus. The two are the same 441 with no filter on; under
            * a filter they are not, and "of 441" would claim a search
            * of the whole collection that never happened.
            */}
          <p className="count">
            {outcome.results.length} of {outcome.pool_size} photographs
          </p>
          <p className="line">{outcome.basis}</p>

          {outcome.low_confidence && (
            <p className="caution">
              No caption mentions these terms and neither score curve
              flattened. These are nearest neighbours, not matches — treat
              the count as a ceiling, not an answer.
            </p>
          )}
        </div>
      )}

      {/* The grid renders its own failure; this one is the palette's. */}
      {failed && <p className="failed">{failed}</p>}

      {/* The strip always shows the whole collection: it is the claim
          that this is everything, which a search does not change. */}
      {palette && (
        <DensityStrip marks={palette.marks} highlight={highlight} />
      )}

      <PhotoGrid collection={collection} />
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
        <Collection
          person={route.person}
          event={route.event}
          q={route.q}
        />
      )}
    </main>
  )
}
