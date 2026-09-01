import { describe, expect, it } from 'vitest'
import { counted, plural } from './plural'

/**
 * An event holding one photograph rendered as "1 images" on the item
 * page. The count is composed in several templates here, so the
 * agreement lives in one function that they all call.
 */

describe('plural', () => {
  it('takes the singular only at one', () => {
    // Zero is plural in English: "0 images", not "0 image".
    expect(plural(0, 'image')).toBe('images')
    expect(plural(1, 'image')).toBe('image')
    expect(plural(2, 'image')).toBe('images')
  })

  it('accepts an irregular plural', () => {
    expect(plural(1, 'person', 'people')).toBe('person')
    expect(plural(3, 'person', 'people')).toBe('people')
  })
})

describe('counted', () => {
  it('puts the number with the noun that agrees', () => {
    expect(counted(1, 'image')).toBe('1 image')
    expect(counted(13, 'image')).toBe('13 images')
  })

  it('composes a split phrase against the right number', () => {
    // "3 of 4 photographs" agrees with the total, not with the count
    // in front of it, which is why plural() returns the noun alone.
    expect(`3 of ${counted(4, 'photograph')}`).toBe('3 of 4 photographs')
    expect(`1 of ${counted(1, 'photograph')}`).toBe('1 of 1 photograph')
  })

  it('renders the event line that was reported', () => {
    expect(` · ${counted(1, 'image')}`).toBe(' · 1 image')
    expect(` · ${counted(13, 'image')}`).toBe(' · 13 images')
  })
})
