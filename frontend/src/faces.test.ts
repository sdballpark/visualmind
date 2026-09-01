import { describe, expect, it } from 'vitest'
import { unmatchedNote } from './faces'

/**
 * The item page listed the people clustering could place and said
 * nothing about the faces it could not, so a photograph with three
 * faces and two names read as a photograph with two people.
 */

describe('unmatchedNote', () => {
  it('says nothing when every face was placed', () => {
    // 232 of the 365 photographs with a face in them are in this case.
    expect(unmatchedNote(0)).toBeNull()
  })

  it('agrees in the singular', () => {
    expect(unmatchedNote(1)).toBe(
      '1 face detected but not matched to anyone',
    )
  })

  it('agrees in the plural', () => {
    expect(unmatchedNote(2)).toBe(
      '2 faces detected but not matched to anyone',
    )
    expect(unmatchedNote(11)).toBe(
      '11 faces detected but not matched to anyone',
    )
  })

  it('names nobody and scores nothing', () => {
    // Clustering declined to place the face. A name or a distance
    // beside it would invite the trust the rejection withheld - the
    // same error as a search presenting a gradient guess as a match.
    const note = unmatchedNote(1) ?? ''

    expect(note).not.toMatch(/[0-9]\.[0-9]/)
    expect(note.toLowerCase()).not.toContain('nearest')
    expect(note.toLowerCase()).not.toContain('probably')
  })

  it('treats a negative count as nothing to say', () => {
    expect(unmatchedNote(-1)).toBeNull()
  })
})
