import { beforeEach, describe, expect, it } from 'vitest'
import { forgetResults, recallResult, remember } from './searchMemory'
import type { SearchResponse, SearchResult } from './api'

/**
 * What the item page can say about a photograph it was sent to.
 *
 * The ranks belong to a search, not to an image, so the panel has to be
 * able to tell "this was rank 4" from "no search brought you here".
 * Conflating them would put zeroes on a page nobody searched for.
 */

function result(over: Partial<SearchResult>): SearchResult {
  return {
    sha256: 'sha-1',
    filename: 'one.jpg',
    caption: 'a dog',
    captured: null,
    grid: null,
    lightbox: null,
    rank: 1,
    score: 0.5,
    image_rank: 10,
    caption_rank: 1,
    term_hits: 1,
    matched: true,
    ...over,
  }
}

function outcome(results: SearchResult[]): SearchResponse {
  return {
    results,
    basis: 'Every caption here mentions the query term.',
    basis_kind: 'full_match',
    score_kind: 'caption_cosine',
    low_confidence: false,
    total_terms: 1,
    full_count: results.length,
    partial_count: 0,
    people: [],
    events: [],
    pool_size: 441,
    corpus_size: 441,
    understood: {
      query: 'dog', persons: [], events: [], terms: 'dog',
      dropped: [], source: 'fallback', note: 'no person or event named',
    },
  }
}

beforeEach(() => forgetResults())

describe('recallResult', () => {
  it('knows nothing before a search', () => {
    expect(recallResult('sha-1')).toBeNull()
  })

  it('returns what the search knew about a result', () => {
    remember('dog', outcome([result({})]))

    const found = recallResult('sha-1')

    expect(found).not.toBeNull()
    expect(found!.query).toBe('dog')
    expect(found!.scoreKind).toBe('caption_cosine')
    expect(found!.totalTerms).toBe(1)
    expect(found!.result.image_rank).toBe(10)
    expect(found!.result.caption_rank).toBe(1)
  })

  it('is null for an image the search did not return', () => {
    // A deep link, or a click from the unfiltered grid. The panel says
    // so rather than showing a rank of zero, which would be a claim.
    remember('dog', outcome([result({})]))

    expect(recallResult('sha-elsewhere')).toBeNull()
  })

  it('keeps a null rank null rather than defaulting it', () => {
    // Either modality can decline to rank an image, and "not ranked" is
    // a different fact from "ranked last".
    remember('dog', outcome([result({ image_rank: null })]))

    expect(recallResult('sha-1')!.result.image_rank).toBeNull()
  })

  it('replaces the previous search rather than accumulating', () => {
    // Only the search the reader actually came from is true of the
    // page they are on; keeping the older one would attach a stale
    // rank to an image the new query never returned.
    remember('dog', outcome([result({ sha256: 'sha-old' })]))
    remember('beach', outcome([result({ sha256: 'sha-new', rank: 4 })]))

    expect(recallResult('sha-old')).toBeNull()
    expect(recallResult('sha-new')!.query).toBe('beach')
    expect(recallResult('sha-new')!.result.rank).toBe(4)
  })

  it('carries the term count the query was measured against', () => {
    remember(
      'a red car',
      { ...outcome([result({ term_hits: 2 })]), total_terms: 2 },
    )

    const found = recallResult('sha-1')!

    expect(found.totalTerms).toBe(2)
    expect(found.result.term_hits).toBe(2)
  })
})
