import { describe, expect, it } from 'vitest'
import { parseNotes } from './parseNotes'
import type { Understood } from './api'

function understood(over: Partial<Understood>): Understood {
  return {
    query: 'dog',
    persons: [],
    events: [],
    terms: 'dog',
    dropped: [],
    source: 'fallback',
    note: 'no person or event named',
    ...over,
  }
}

const kinds = (u: Understood) => parseNotes(u).map((n) => n.kind)
const texts = (u: Understood) =>
  parseNotes(u).map((n) => ('text' in n ? n.text : `${n.label}: ${n.value}`))

describe('a query that named nobody', () => {
  it('says nothing at all', () => {
    // The common case. A line on every descriptive search would bury
    // the two that matter.
    expect(parseNotes(understood({}))).toEqual([])
  })

  it('says nothing when understanding was turned off', () => {
    expect(parseNotes(understood({
      note: 'understanding disabled for this request',
    }))).toEqual([])
  })
})

describe('a query that named someone', () => {
  it('reports the person and what was left to search', () => {
    expect(texts(understood({
      query: 'Bob with sunglasses',
      persons: ['Bob Welch'],
      terms: 'with sunglasses',
      source: 'model',
      note: '',
    }))).toEqual([
      'Person: Bob Welch',
      'Searched for: with sunglasses',
    ])
  })

  it('says so when the name was the whole query', () => {
    expect(texts(understood({
      query: 'Andrew Bogan',
      persons: ['Andrew Bogan'],
      terms: '',
      source: 'model',
      note: '',
    }))).toEqual([
      'Person: Andrew Bogan',
      'Searched for: nothing else',
    ])
  })

  it('agrees in number and joins several names', () => {
    expect(texts(understood({
      query: 'Daniel and Casey',
      persons: ['Daniel Bogan', 'Casey Bogan'],
      terms: 'and',
      source: 'model',
      note: '',
    }))[0]).toBe('People: Daniel Bogan and Casey Bogan')
  })

  it('does not restate the terms when nothing was taken out', () => {
    expect(texts(understood({
      query: 'Bob', persons: ['Bob Welch'], terms: 'Bob',
      source: 'model', note: '',
    }))).toEqual(['Person: Bob Welch'])
  })
})

describe('a rejection', () => {
  it('is stated, with the reason', () => {
    // The _cartoon case: a name that resolved against the roster and
    // was never in the query. This is the line the layer exists for.
    const notes = parseNotes(understood({
      query: 'people wearing sunglasses',
      terms: 'people wearing sunglasses',
      dropped: [
        { text: '_cartoon', as: 'person', why: 'not named in the query' },
      ],
      source: 'model',
      note: 'nothing proposed survived validation',
    }))

    expect(notes).toHaveLength(1)
    expect(notes[0].kind).toBe('rejected')
    expect(texts(understood({
      dropped: [
        { text: '_cartoon', as: 'person', why: 'not named in the query' },
      ],
      source: 'model',
    }))[0]).toBe('Ignored “_cartoon” — the query does not mention it.')
  })

  it('explains each reason in words', () => {
    const why = (reason: string) =>
      texts(understood({
        dropped: [{ text: 'X', as: 'person', why: reason }],
        source: 'model',
      }))[0]

    expect(why('unknown')).toContain('nobody by that name is labelled')
    expect(why('ambiguous')).toContain('more than one person')
    expect(why('too many to be a filter')).toContain('too many')
  })

  it('falls back to the raw reason rather than dropping the note', () => {
    // A reason this file has not met is still worth showing.
    expect(texts(understood({
      dropped: [{ text: 'X', as: 'person', why: 'something new' }],
      source: 'model',
    }))[0]).toBe('Ignored “X” — something new.')
  })

  it('is reported even when a name was also kept', () => {
    expect(kinds(understood({
      query: 'Bob with sunglasses',
      persons: ['Bob Welch'],
      terms: 'with sunglasses',
      dropped: [{ text: 'Nobody', as: 'person', why: 'unknown' }],
      source: 'model',
      note: '',
    }))).toEqual(['read', 'read', 'rejected'])
  })
})

describe('an unavailable model', () => {
  it('is distinguishable from a query that named nobody', () => {
    // Both search the whole query as text. Only one of them is a
    // statement about the query.
    expect(kinds(understood({ note: 'model unavailable' })))
      .toEqual(['unavailable'])
    expect(kinds(understood({ note: 'no person or event named' })))
      .toEqual([])
  })

  it('separates not loading from not being configured', () => {
    expect(texts(understood({ note: 'model unavailable' }))[0])
      .toContain('did not load')
    expect(texts(understood({
      note: 'no model registered for query_understanding',
    }))[0]).toContain('No query model is configured')
  })

  it('reports a reply it could not use', () => {
    expect(texts(understood({
      note: 'model returned no usable JSON',
    }))[0]).toContain('nothing usable')
  })

  it('is not claimed when the model did answer', () => {
    expect(kinds(understood({
      source: 'model', note: 'model unavailable',
    }))).toEqual([])
  })
})

describe('robustness', () => {
  it('says nothing when there is no parse at all', () => {
    expect(parseNotes(null)).toEqual([])
  })
})
