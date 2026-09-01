import { counted } from './plural'

/**
 * What to say about faces clustering could not place with anyone.
 *
 * `people` lists the faces it could place, so a photograph with three
 * detected faces and two named people otherwise reads as a photograph
 * with two people. This is the sentence that stops it doing that - the
 * same shape of admission as the strip's undated segment, the basis
 * line's gradient guess, and the diagnostics panel saying no search
 * brought you here.
 *
 * A count and nothing else. Every unplaced face has a nearest labelled
 * neighbour, and naming it would invite the trust the clusterer withheld
 * when it declined to place the face.
 *
 * Null at zero: a photograph where every detected face was placed has
 * nothing to admit, and a line saying so would be noise on 232 of the
 * 365 photographs that contain a face at all.
 */
export function unmatchedNote(count: number): string | null {
  if (count <= 0) {
    return null
  }

  return `${counted(count, 'face')} detected but not matched to anyone`
}
