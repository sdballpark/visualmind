import { beforeEach, describe, expect, it } from 'vitest'
import { forget, parseRoute, pathFor, recall, remember } from './router'

const SHA = 'a'.repeat(64)

describe('parseRoute', () => {
  it('reads a photo route', () => {
    expect(parseRoute(`/photo/${SHA}`)).toEqual({ name: 'photo', sha: SHA })
  })

  it('treats the root as the collection', () => {
    expect(parseRoute('/')).toEqual({ name: 'grid' })
  })

  it('refuses anything that is not a sha', () => {
    // A sha is 64 hex characters. Nothing else addresses a photograph,
    // so anything else is the collection rather than a broken page.
    expect(parseRoute('/photo/nonsense')).toEqual({ name: 'grid' })
    expect(parseRoute('/photo/' + 'a'.repeat(63))).toEqual({ name: 'grid' })
    expect(parseRoute('/photo/' + 'A'.repeat(64))).toEqual({ name: 'grid' })
    expect(parseRoute('/photo/' + SHA + '/extra')).toEqual({ name: 'grid' })
  })

  it('round-trips through pathFor', () => {
    const route = { name: 'photo', sha: SHA } as const

    expect(parseRoute(pathFor(route))).toEqual(route)
    expect(parseRoute(pathFor({ name: 'grid' }))).toEqual({ name: 'grid' })
  })
})

describe('scroll memory', () => {
  beforeEach(() => forget('/'))

  it('is zero for a route never visited', () => {
    // Not undefined: the grid compares it against its own height, and a
    // missing value would restore to NaN.
    expect(recall('/')).toBe(0)
  })

  it('remembers where the reader left a route', () => {
    remember('/', 4820)

    expect(recall('/')).toBe(4820)
  })

  it('keeps routes apart', () => {
    remember('/', 4820)
    remember(`/photo/${SHA}`, 0)

    expect(recall('/')).toBe(4820)
  })
})
