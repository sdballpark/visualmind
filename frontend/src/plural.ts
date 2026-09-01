/**
 * Numbers rendered for a reader.
 *
 * An event holding one photograph rendered as "1 images" on the item
 * page. The count is composed in more than one place here, so the
 * agreement lives in one function rather than in each template string.
 */

/**
 * The form of `singular` that agrees with `count`.
 *
 * Returns the noun alone rather than the number with it, because the
 * two are not always adjacent: "3 of 4 photographs" agrees with the
 * total, not with the count standing in front of it.
 */
export function plural(count: number, singular: string, many?: string): string {
  return count === 1 ? singular : (many ?? `${singular}s`)
}

/** `count` followed by the noun that agrees with it. */
export function counted(
  count: number,
  singular: string,
  many?: string,
): string {
  return `${count} ${plural(count, singular, many)}`
}
