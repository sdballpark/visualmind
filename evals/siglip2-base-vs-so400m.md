# SigLIP2 base-224 vs SO400M-384 on a personal photo corpus

## Setup

- Corpus: 441 family photos, mixed 2002–2026, mostly consumer digicam JPEGs and scans
- Models: `google/siglip2-base-patch16-224` vs `google/siglip2-so400m-patch14-384`
- Identical pipeline; only the registry entry changed between runs
- Manual relevance labelling of the top 12 results per query

## Result

| Query         | base-224 | SO400M-384 |
|---------------|----------|------------|
| "dog"         | 10 / 12  | 5 / 12     |
| "dog indoors" | 7 / 12   | 4 / 12     |

The larger, higher-resolution model performed worse on both queries.

## Interpretation

The likely cause is source resolution. Much of this corpus predates 2010 and
carries heavy JPEG artifacts. At 384px the input is upsampled rather than
better resolved, so the additional capacity appears to fit compression noise
instead of content. Base at 224px is closer to the effective resolution of the
material.

Caveat: two queries over a corpus containing roughly eight relevant images is
a small sample. The direction is consistent but the magnitude is not precise.

## Secondary finding

Compound queries underperform bare object nouns at this model size. "dog"
(10/12) beat "dog indoors" (7/12) on the same corpus — adding a scene
qualifier appears to split the embedding between concepts rather than
narrowing the result set.

## Decision

Reverted to base-224. SO400M is retained in the registry as
`image_embedding_alternate` for reproducibility.

## Not yet tested

- Whether the gap holds on the post-2015 subset, where resolution is adequate
- Whether captioning + text search sidesteps the compound-query weakness
