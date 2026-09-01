import { useEffect, useState } from 'react'

/**
 * Two routes, so a hand-rolled router rather than a dependency.
 *
 * The reason it is worth writing out is scroll restoration. The grid is
 * virtualized and pre-allocates its height from a total that arrives
 * with the first page, so a scroll position restored before that height
 * exists gets clamped to a short document and the reader lands at the
 * top. Position is therefore remembered here and replayed by the grid
 * once it has the height to hold it.
 */

export type Route =
  | { name: 'grid'; person?: string; event?: string; q?: string }
  | { name: 'photo'; sha: string }

const SHA = /^\/photo\/([0-9a-f]{64})$/

export function parseRoute(pathname: string, search = ''): Route {
  const match = SHA.exec(pathname)

  if (match) {
    return { name: 'photo', sha: match[1] }
  }

  const params = new URLSearchParams(search)

  // An empty q is dropped rather than carried as "". The API rejects a
  // request with no query and no filter, and "/?q=" would otherwise be
  // a URL that always errors.
  const q = params.get('q')?.trim()

  return {
    name: 'grid',
    person: params.get('person') ?? undefined,
    event: params.get('event') ?? undefined,
    q: q ? q : undefined,
  }
}

export function pathFor(route: Route): string {
  if (route.name === 'photo') {
    return `/photo/${route.sha}`
  }

  const params = new URLSearchParams()

  if (route.person) {
    params.set('person', route.person)
  }

  if (route.event) {
    params.set('event', route.event)
  }

  if (route.q) {
    params.set('q', route.q)
  }

  const query = params.toString()

  return query ? `/?${query}` : '/'
}

const remembered = new Map<string, number>()

export function currentPath(): string {
  return window.location.pathname + window.location.search
}

/** Where the reader was on a route they may come back to. */
export function remember(path: string, scrollTop: number) {
  remembered.set(path, scrollTop)
}

export function recall(path: string): number {
  return remembered.get(path) ?? 0
}

export function forget(path: string) {
  remembered.delete(path)
}

export function navigate(to: string) {
  remember(currentPath(), window.scrollY)
  window.history.pushState({}, '', to)
  window.dispatchEvent(new PopStateEvent('popstate'))
  window.scrollTo(0, 0)
}

export function useRoute(): Route {
  const [route, setRoute] = useState<Route>(() =>
    parseRoute(window.location.pathname, window.location.search),
  )

  useEffect(() => {
    const read = () =>
      setRoute(parseRoute(window.location.pathname, window.location.search))

    window.addEventListener('popstate', read)

    return () => window.removeEventListener('popstate', read)
  }, [])

  return route
}
