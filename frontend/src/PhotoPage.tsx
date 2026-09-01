import { useCallback, useEffect, useState } from 'react'
import {
  fetchImage,
  fetchImages,
  gridThumbnail,
  lightboxImage,
  type ImageDetail,
  type ImageRecord,
} from './api'
import { navigate } from './router'

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

function Related({ detail }: { detail: ImageDetail }) {
  const { people, event, duplicates } = detail
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
      </dd>

      <dt>Event</dt>
      <dd>
        {event ? (
          <a
            href={`/?event=${encodeURIComponent(event.id)}`}
            onClick={(clicked) => {
              clicked.preventDefault()
              navigate(`/?event=${encodeURIComponent(event.id)}`)
            }}
          >
            {event.name}
            <span className="aside">
              {event.start ? ` · ${event.start.slice(0, 10)}` : ''}
              {` · ${event.images} images`}
            </span>
          </a>
        ) : (
          <span className="absent">unassigned</span>
        )}
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

          <dt>retrieval</dt>
          <dd className="absent">
            image_rank, caption_rank, score_kind and term hits are
            properties of a search result. This page was not reached from
            one, so there are none to show.
          </dd>
        </dl>
      )}
    </section>
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
