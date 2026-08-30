# Retrieval evaluation: SigLIP2 on a personal photo corpus

## Setup

- Corpus: 441 family photos, mixed 2002-2026, mostly consumer digicam JPEGs and scans
- Primary model: `google/siglip2-base-patch16-224`
- Compared against: `google/siglip2-so400m-patch14-384`
- Identical pipeline; only the registry entry changed between runs
- Manual relevance labelling of the top 12 results per query

## Finding 1 - the larger model performed worse

| Query         | base-224 | SO400M-384 |
|---------------|----------|------------|
| "dog"         | 10 / 12  | 5 / 12     |
| "dog indoors" | 7 / 12   | 4 / 12     |

Likely cause is source resolution. Much of this corpus predates 2010 and
carries heavy JPEG artifacts. At 384px the input is upsampled rather than
better resolved, so the additional capacity appears to fit compression
artifacts instead of content.

Reverted to base-224. SO400M is retained in the registry as
`image_embedding_alternate` for reproducibility.

## Finding 2 - accuracy splits by concept scale, not query phrasing

| Query                    | Concept type | Result  |
|--------------------------|--------------|---------|
| "beach"                  | scene        | 12 / 12 |
| "people at the beach"    | scene        | 12 / 12 |
| "dog"                    | small object | 10 / 12 |
| "dog indoors"            | small object | 7 / 12  |
| "christmas tree indoors" | small object | 5 / 12  |
| "christmas tree"         | small object | 4 / 12  |

Whole-frame scenes retrieve reliably; objects occupying a small fraction of
the frame degrade sharply. At 224px a dog or a tree in a family snapshot is
a few dozen pixels.

An earlier hypothesis that compound queries ("X indoors") underperform bare
nouns did not survive testing. Compound phrasing helped for Christmas, hurt
for dogs, and was neutral for beach. That hypothesis is rejected.

## Finding 3 - absent concepts are indistinguishable by score alone

"cat" was run as a control; the corpus contains no cats. Top-12 scores:

```
0.0929 0.0908 0.0890 0.0856 0.0851 0.0840
0.0836 0.0834 0.0822 0.0808 0.0783 0.0782
```

The top score for a concept with zero instances (0.0929) sits within 12% of
the top score for a concept that is present ("dog", 0.1046). A fixed
similarity floor therefore cannot separate present from absent concepts -
any threshold that excludes cats also excludes dogs.

The distinguishing signal is the shape of the decay, not the absolute value:

| Query | Rank 1 | Rank 12 | Spread |
|-------|--------|---------|--------|
| "dog" | 0.1046 | 0.0531  | 0.0515 |
| "cat" | 0.0929 | 0.0782  | 0.0147 |

A present concept produces a steep curve. An absent one produces a plateau of
roughly equidistant nearest neighbours. Retrieval confidence should be derived
from spread or gradient rather than from a fixed cutoff.

## Note on SigLIP score scale

SigLIP is trained with a sigmoid loss and a learned temperature/bias, so raw
cosine similarities cluster near zero rather than in the 0.2-0.35 band typical
of CLIP. A score of 0.10 here is a strong match, not a weak one.

## Implications

- A bigger embedding model is not the fix for small-object retrieval - that was
  tested and regressed.
- The remaining paths for small objects are detection (crop, then embed) or
  captioning (describe, then search text).
- `--min-score` as currently implemented is of limited value and should be
  supplemented by a spread-based confidence measure.

## Not yet tested

- Whether the resolution gap holds on the post-2015 subset
- Whether captioning sidesteps the small-object weakness
- Spread thresholds across a wider query set
