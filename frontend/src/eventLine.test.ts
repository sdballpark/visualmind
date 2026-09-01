import { describe, expect, it } from 'vitest'
import { eventLine } from './eventLine'
import type { EventSummary } from './api'

function event(over: Partial<EventSummary>): EventSummary {
  return {
    id: 'event-001',
    name: 'Dec 2002',
    start: '2002-12-25',
    end: '2002-12-25',
    images: 4,
    ...over,
  }
}

describe('eventLine', () => {
  it('links an event that holds more than one photograph', () => {
    const line = eventLine(event({ images: 32 }))

    expect(line.kind).toBe('event')
  })

  it('still links at two', () => {
    // Two photographs from one afternoon is a real occasion, and the
    // link goes somewhere. The argument was never about size.
    expect(eventLine(event({ images: 2 })).kind).toBe('event')
  })

  it('does not link a singleton', () => {
    // The link would return the reader to a grid holding the one
    // photograph they are already looking at.
    const line = eventLine(event({ images: 1 }))

    expect(line.kind).toBe('alone')
  })

  it('keeps a singleton name and date', () => {
    // "Dec 2002" with thirteen people in it is a real occasion; the
    // corpus holding one photograph of it does not make it less real.
    const line = eventLine(event({ images: 1 }))

    expect(line).toEqual({
      kind: 'alone',
      name: 'Dec 2002',
      date: '2002-12-25',
    })
  })

  it('drops the count from a singleton', () => {
    // "1 image" was a tautology: this photograph is one photograph.
    const line = eventLine(event({ images: 1 }))

    expect(JSON.stringify(line)).not.toContain('image')
  })

  it('survives an event with no start date', () => {
    expect(eventLine(event({ images: 1, start: null }))).toEqual({
      kind: 'alone',
      name: 'Dec 2002',
      date: null,
    })
  })

  it('reports no event at all as absent', () => {
    // The API returns null for the unassigned bucket too, because
    // being undated is not an occasion 118 photographs share.
    expect(eventLine(null)).toEqual({ kind: 'none' })
  })

  it('treats a zero-image event as alone rather than as a link', () => {
    // Should not arise, but a link to an empty grid is the worse of
    // the two failures.
    expect(eventLine(event({ images: 0 })).kind).toBe('alone')
  })
})
