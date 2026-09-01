import type { EventSummary } from './api'

/**
 * What the item page should say about this photograph's event.
 *
 * Three cases, because an event of one is a different kind of thing
 * from an event of thirty-two, and rendering them identically is what
 * produced "1 images".
 *
 * A singleton keeps its name and date - "Dec 2002" with thirteen people
 * in it is a real occasion, and the corpus holding one photograph of it
 * does not make it less real. What it loses is the link and the count.
 * The count was a tautology: the line said this photograph, taken when
 * it was taken, is one photograph. The link was worse - the item page
 * exists to make every relationship a way to leave, and for 40% of the
 * events in this corpus it led to a grid holding only the photograph
 * you were already looking at.
 *
 * `null` is the absence of an event, which the API now also returns for
 * the unassigned bucket. That bucket is where an image goes when it has
 * no capture time and no unambiguous thread, so it is not an occasion
 * 118 photographs share.
 */
export type EventLine =
  | { kind: 'none' }
  | { kind: 'alone'; name: string; date: string | null }
  | { kind: 'event'; event: EventSummary; date: string | null }

function dateOf(event: EventSummary): string | null {
  return event.start ? event.start.slice(0, 10) : null
}

export function eventLine(event: EventSummary | null): EventLine {
  if (!event) {
    return { kind: 'none' }
  }

  if (event.images <= 1) {
    return { kind: 'alone', name: event.name, date: dateOf(event) }
  }

  return { kind: 'event', event, date: dateOf(event) }
}
