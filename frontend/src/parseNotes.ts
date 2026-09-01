import type { Understood } from './api'

/**
 * What to say about how the query was read.
 *
 * Sits with the basis line because it is the same kind of admission: the
 * basis says on what evidence these results were chosen, and this says
 * what the system thought the question was. A reader who types "Bob with
 * sunglasses" should be able to see that it became a person and a term.
 *
 * The rejections carry the most weight of anything here. A model asked
 * for names against a roster will sometimes return one that resolves and
 * was never asked for - against the real corpus, "people wearing
 * sunglasses" produced a roster entry called "_cartoon" - and the layer
 * exists to stop that reaching retrieval. Stopping it silently would
 * leave the reader with a correct answer and no way to know how close
 * they came to a confidently wrong one, so a rejection is stated and its
 * reason with it.
 *
 * A query that simply named nobody says nothing at all. That is the
 * common case, and a line reporting it on every descriptive search would
 * bury the two that matter.
 */
export type ParseNote =
  | { kind: 'read'; label: string; value: string }
  | { kind: 'rejected'; text: string }
  | { kind: 'unavailable'; text: string }

/**
 * Fallback notes worth showing. The rest - a query that named nobody, an
 * empty query, an explicit opt-out - are silence: the reader learns
 * nothing from being told that a search of their words searched their
 * words.
 */
const UNAVAILABLE: Record<string, string> = {
  'model unavailable':
    'The query model did not load, so the whole query was searched as text.',
  'no model registered for query_understanding':
    'No query model is configured, so the whole query was searched as text.',
  'model returned no usable JSON':
    'The query model returned nothing usable, so the whole query was '
    + 'searched as text.',
}

const WHY: Record<string, string> = {
  unknown: 'nobody by that name is labelled in this collection',
  ambiguous: 'it matches more than one person',
  'not named in the query': 'the query does not mention it',
  'too many to be a filter': 'too many to be a filter',
}

function list(values: string[]): string {
  if (values.length <= 1) {
    return values.join('')
  }

  return values.slice(0, -1).join(', ') + ' and ' + values[values.length - 1]
}

export function parseNotes(understood: Understood | null): ParseNote[] {
  if (!understood) {
    return []
  }

  const notes: ParseNote[] = []

  if (understood.persons.length > 0) {
    notes.push({
      kind: 'read',
      label: understood.persons.length === 1 ? 'Person' : 'People',
      value: list(understood.persons),
    })
  }

  if (understood.events.length > 0) {
    notes.push({
      kind: 'read',
      label: understood.events.length === 1 ? 'Event' : 'Events',
      value: list(understood.events),
    })
  }

  // Only worth stating when it is not simply the query back again.
  if (
    (understood.persons.length > 0 || understood.events.length > 0) &&
    understood.terms !== understood.query
  ) {
    notes.push({
      kind: 'read',
      label: 'Searched for',
      value: understood.terms || 'nothing else',
    })
  }

  for (const drop of understood.dropped) {
    notes.push({
      kind: 'rejected',
      text:
        `Ignored “${drop.text}” — ${WHY[drop.why] ?? drop.why}.`,
    })
  }

  const unavailable = UNAVAILABLE[understood.note]

  if (understood.source === 'fallback' && unavailable) {
    notes.push({ kind: 'unavailable', text: unavailable })
  }

  return notes
}
