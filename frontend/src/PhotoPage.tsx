import { useCallback, useEffect, useState } from 'react'
import {
  fetchImage,
  fetchImages,
  gridThumbnail,
  lightboxImage,
  type ImageDetail,
  type ImageRecord,
} from './api'
import { eventLine, type EventLine } from './eventLine'
import { unmatchedNote } from './faces'
import { counted } from './plural'
import { navigate } from './router'
import { recallResult } from './searchMemory'

/**
 * One photograph, at its own URL.
 *
 * Not a lightbox: a deep-linkable page that shows what the system knows
 * about this image and makes every relationship a way to leave. The
 * strip along the bottom carries the surrounding catalog so the reader
 * can move sideways without going back first.
 */

/** How many neighbours either side to pull for the strip. */
const REACH = 14

/**
 * A singleton is not linked. The link would return the reader to a grid
 * holding the one photograph they are already on, which is a door onto
 * the room they are standing in rather than a way out.
 */
function Event({ line }: { line: EventLine }) {
  if (line.kind === 'none') {
    return <span className="absent">not part of any event</span>
  }

  if (line.kind === 'alone') {
    return (
      <span>
        {line.name}
        <span className="aside">
          {line.date ? ` · ${line.date}` : ''}
          {' · the only photograph from it'}
        </span>
      </span>
    )
  }

  const { event, date } = line

  return (
    <a
      href={`/?event=${encodeURIComponent(event.id)}`}
      onClick={(clicked) => {
        clicked.preventDefault()
        navigate(`/?event=${encodeURIComponent(event.id)}`)
      }}
    >
      {event.name}
      <span className="aside">
        {date ? ` · ${date}` : ''}
        {` · ${counted(event.images, 'image')}`}
      </span>
    </a>
  )
}


function Related({ detail }: { detail: ImageDetail }) {
  const { people, event, duplicates } = detail
  const unmatched = unmatchedNote(detail.unmatched_faces)
  const siblings = duplicates?.members.filter(
    (member) => member.sha256 !== detail.sha256,
  )

  return (
    <dl className="related">
      <dt>People</dt>
      <dd>
        {people.length === 0 && <span className="absent">none detected</span>}
        {people.map((person) => (
          <a
            key={person.name}
            className="person"
            href={`/?person=${encodeURIComponent(person.name)}`}
            onClick={(clicked) => {
              clicked.preventDefault()
              navigate(`/?person=${encodeURIComponent(person.name)}`)
            }}
          >
            {person.name}
          </a>
        ))}
        {/* Beneath the names, in the register the absences use. */}
        {unmatched && <span className="unmatched">{unmatched}</span>}
      </dd>

      <dt>Event</dt>
      <dd>
        <Event line={eventLine(event)} />
      </dd>

      <dt>Near-duplicates</dt>
      <dd>
        {siblings && siblings.length > 0 ? (
          <span className="siblings">
            {siblings.map((member) => (
              <a
                key={member.sha256}
                href={`/photo/${member.sha256}`}
                onClick={(event_) => {
                  event_.preventDefault()
                  navigate(`/photo/${member.sha256}`)
                }}
              >
                <img src={gridThumbnail(member.sha256)} alt={member.filename} />
              </a>
            ))}
            <span className="aside">{duplicates?.tier.toLowerCase()}</span>
          </span>
        ) : (
          <span className="absent">none found</span>
        )}
      </dd>
    </dl>
  )
}

function Diagnostics({ detail }: { detail: ImageDetail }) {
  const [open, setOpen] = useState(false)

  return (
    <section className="diagnostics">
      <button type="button" onClick={() => setOpen(!open)}>
        {open ? 'Hide diagnostics' : 'Diagnostics'}
      </button>

      {open && (
        <dl>
          <dt>sha256</dt>
          <dd className="mono">{detail.sha256}</dd>

          <dt>catalog position</dt>
          <dd>{detail.index + 1} of {detail.total}</dd>

          <dt>captured</dt>
          <dd>{detail.captured ?? 'no EXIF date'}</dd>

          <dt>grid</dt>
          <dd>
            {detail.grid
              ? `${detail.grid.width} × ${detail.grid.height}`
              : 'no thumbnail'}
          </dd>

          <dt>lightbox</dt>
          <dd>
            {detail.lightbox
              ? `${detail.lightbox.width} × ${detail.lightbox.height}`
              : 'no thumbnail'}
          </dd>

          <dt>duplicate group</dt>
          <dd>
            {detail.duplicates
              ? `${detail.duplicates.group} · ${detail.duplicates.tier}`
              : 'none'}
          </dd>

          <Retrieval sha={detail.sha256} />
        </dl>
      )}
    </section>
  )
}

/** A rank the other modality did not produce is absent, not zero. */
function rank(value: number | null): string {
  return value === null ? 'not ranked' : `#${value}`
}

/**
 * What the search that led here knew about this photograph.
 *
 * Absent when there was no such search - a deep link, a reload, or a
 * click from the unfiltered grid. That case keeps its original wording
 * rather than showing zeroes, because a rank of none and a rank of
 * nothing-was-asked are different facts and only one of them is true.
 */
function Retrieval({ sha }: { sha: string }) {
  const context = recallResult(sha)

  if (!context) {
    return (
      <>
        <dt>retrieval</dt>
        <dd className="absent">
          image_rank, caption_rank, score_kind and term hits are
          properties of a search result. This page was not reached from
          one, so there are none to show.
        </dd>
      </>
    )
  }

  const { query, scoreKind, totalTerms, result } = context

  return (
    <>
      <dt>from search</dt>
      <dd>
        <span className="query">{query}</span>
        {result.matched ? ' · term match' : ' · not a term match'}
      </dd>

      <dt>rank</dt>
      <dd>{result.rank}</dd>

      <dt>image_rank</dt>
      <dd>{rank(result.image_rank)}</dd>

      <dt>caption_rank</dt>
      <dd>{rank(result.caption_rank)}</dd>

      <dt>term hits</dt>
      <dd>
        {result.term_hits} of {totalTerms}
        {totalTerms === 1 ? ' term' : ' terms'}
      </dd>

      <dt>score</dt>
      <dd>
        {result.score.toFixed(4)}
        {/*
          * The scale is named because the three are an order of
          * magnitude apart and nothing in the number says which it is:
          * an RRF sum sits near 0.03 where a BGE cosine sits near 0.7.
          */}
        <span className="aside"> · {scoreKind}</span>
      </dd>
    </>
  )
}

function Strip({ detail }: { detail: ImageDetail }) {
  const [around, setAround] = useState<ImageRecord[]>([])

  useEffect(() => {
    const offset = Math.max(0, detail.index - REACH)

    fetchImages(offset, REACH * 2 + 1)
      .then((page) => setAround(page.images))
      .catch(() => setAround([]))
  }, [detail.index])

  const position = around.findIndex((item) => item.sha256 === detail.sha256)

  const step = useCallback(
    (by: number) => {
      const next = around[position + by]

      if (next) {
        navigate(`/photo/${next.sha256}`)
      }
    },
    [around, position],
  )

  useEffect(() => {
    const onKey = (pressed: KeyboardEvent) => {
      if (pressed.key === 'ArrowLeft') {
        step(-1)
      } else if (pressed.key === 'ArrowRight') {
        step(1)
      }
    }

    window.addEventListener('keydown', onKey)

    return () => window.removeEventListener('keydown', onKey)
  }, [step])

  return (
    <nav className="surrounds" aria-label="Surrounding photographs">
      {around.map((item) => (
        <a
          key={item.sha256}
          href={`/photo/${item.sha256}`}
          className={item.sha256 === detail.sha256 ? 'here' : undefined}
          onClick={(event_) => {
            event_.preventDefault()
            navigate(`/photo/${item.sha256}`)
          }}
        >
          <img src={gridThumbnail(item.sha256)} alt={item.filename} />
        </a>
      ))}
    </nav>
  )
}

export function PhotoPage({ sha }: { sha: string }) {
  const [detail, setDetail] = useState<ImageDetail | null>(null)
  const [failed, setFailed] = useState<string | null>(null)

  useEffect(() => {
    setDetail(null)
    setFailed(null)

    fetchImage(sha)
      .then(setDetail)
      .catch((error) => setFailed(String(error.message ?? error)))
  }, [sha])

  if (failed) {
    return (
      <article className="photo-page">
        <p className="failed">{failed}</p>
      </article>
    )
  }

  if (!detail) {
    return <article className="photo-page" />
  }

  return (
    <article className="photo-page">
      <a
        className="back"
        href="/"
        onClick={(event_) => {
          event_.preventDefault()
          window.history.back()
        }}
      >
        ← the collection
      </a>

      <figure>
        <img
          src={lightboxImage(detail.sha256)}
          alt={detail.caption || detail.filename}
          width={detail.lightbox?.width}
          height={detail.lightbox?.height}
        />
      </figure>

      <div className="prose">
        <p>{detail.caption || 'No caption was generated for this image.'}</p>
      </div>

      <Related detail={detail} />
      <Diagnostics detail={detail} />
      <Strip detail={detail} />
    </article>
  )
}
