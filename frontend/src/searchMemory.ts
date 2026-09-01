import type { SearchResponse, SearchResult } from './api'

/**
 * What the last search knew about each image it returned.
 *
 * image_rank, caption_rank, score_kind and the term hits are properties
 * of a search result, not of a photograph. The item page is a route of
 * its own and cannot recover them: re-running the query there would
 * spend a whole search on a paragraph of numbers, and would still be
 * guessing which query the reader arrived from when several could have
 * returned the same image.
 *
 * So the outcome is remembered here when it arrives, keyed by sha, and
 * the item page reads it. Module state, like the remembered scroll
 * positions in router.ts, and with the same consequence: it lives as
 * long as the tab. A photograph reached by a deep link or a reload has
 * no search behind it and gets the panel's absent case, which is the
 * honest answer rather than a stale one.
 */

export interface ResultContext {
  query: string
  scoreKind: string
  totalTerms: number
  result: SearchResult
}

let lastQuery = ''
let lastScoreKind = ''
let lastTotalTerms = 0
let byImage = new Map<string, SearchResult>()

export function remember(query: string, outcome: SearchResponse) {
  lastQuery = query
  lastScoreKind = outcome.score_kind
  lastTotalTerms = outcome.total_terms
  byImage = new Map(outcome.results.map((result) => [result.sha256, result]))
}

export function recallResult(sha256: string): ResultContext | null {
  const result = byImage.get(sha256)

  if (!result) {
    return null
  }

  return {
    query: lastQuery,
    scoreKind: lastScoreKind,
    totalTerms: lastTotalTerms,
    result,
  }
}

/** Test seam: drop what is remembered. */
export function forgetResults() {
  lastQuery = ''
  lastScoreKind = ''
  lastTotalTerms = 0
  byImage = new Map()
}
