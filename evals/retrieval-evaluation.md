# Retrieval evaluation: SigLIP2, VLM captions, and hybrid search

Evaluated against a 441-image personal photo corpus (family photos,
2002-2026, mostly consumer digicam JPEGs and scans). All relevance
judgements are manual.

## Finding 1 - the larger embedding model performed worse

| Query         | SigLIP2 base-224 | SigLIP2 SO400M-384 |
|---------------|------------------|--------------------|
| "dog"         | 10 / 12          | 5 / 12             |
| "dog indoors" | 7 / 12           | 4 / 12             |

Identical pipeline; only the registry entry changed between runs.

Likely cause is source resolution. Much of this corpus predates 2010 and
carries heavy JPEG artifacts. At 384px the input is upsampled rather than
better resolved, so the additional capacity appears to fit compression
artifacts instead of content.

Reverted to base-224. SO400M is retained in the registry as
image_embedding_alternate for reproducibility.

## Finding 2 - most apparent failures were exhausted results

Initial reading of the top-12 rankings suggested that whole-frame scenes
("beach") retrieved reliably while small objects ("dog", "christmas tree")
degraded badly.

Captioning the full corpus with Qwen3-VL made ground-truth counts available
for the first time, and revised that conclusion:

| Concept        | Images in corpus | Found in top-12 |
|----------------|------------------|-----------------|
| dog            | 20               | 10              |
| christmas tree | 4                | 4               |
| cat            | 0 (2 depictions) | 0               |

SigLIP2 retrieved every Christmas tree that exists. The apparent 4/12 score
was the interface padding a 4-image result set out to 12.

The real defect is that top-k always returns k results. Small-object
retrieval is weaker than scene retrieval - the dog result confirms that -
but far less so than the raw scores implied.

The "cat" control is the clearest case: the corpus contains no cats, only a
stone cat statue and a cat graphic on a shirt. The system returned 12
confidently-ranked results anyway.

## Finding 3 - absolute scores cannot signal absence

Top-12 SigLIP2 scores for "cat", a concept with no instances:

    0.0929 0.0908 0.0890 0.0856 0.0851 0.0840
    0.0836 0.0834 0.0822 0.0808 0.0783 0.0782

The top score for an absent concept (0.0929) sits within 12% of the top
score for a present one ("dog", 0.1046). No fixed similarity floor can
separate them.

The distinguishing signal is the shape of the decay:

| Query | Rank 1 | Rank 12 | Spread |
|-------|--------|---------|--------|
| "dog" | 0.1046 | 0.0531  | 0.0515 |
| "cat" | 0.0929 | 0.0782  | 0.0147 |

A present concept produces a steep curve. An absent one produces a plateau
of roughly equidistant nearest neighbours.

## Finding 4 - a compound-query hypothesis, tested and rejected

An early observation that compound queries ("X indoors") underperform bare
nouns did not survive testing:

| Pair                                         | Direction       |
|----------------------------------------------|-----------------|
| "dog" vs "dog indoors"                       | bare better     |
| "christmas tree" vs "christmas tree indoors" | compound better |
| "beach" vs "people at the beach"             | no difference   |

Rejected. The variation was sample noise over small result sets.

## Finding 5 - captions add recall

Qwen3-VL-4B captioned all 441 images in 23 minutes (3.2s per image, 9.14 GB
peak VRAM with max_pixels capped at 802,816 - the corpus contains 45 MP
images that would otherwise exhaust a 16 GB card).

Captions surface detail no image embedding exposes: text read from signs and
clothing, background objects, counts of people. They also convert fuzzy
ranking into exact matching, which is what made the ground-truth counts in
Finding 2 possible.

Caption retrieval also recovers images the visual index buries. For "cat",
the stone-statue photo sat at image rank 388 of 441 and caption rank 3.

Hybrid search fuses the two rankings with Reciprocal Rank Fusion (k=60),
since the score scales are not comparable - SigLIP sits near 0.10, BGE near
0.7.

## Finding 6 - a derived result count outperforms a fixed one

Rather than always returning k results, the search derives how many to
return from two signals:

1. Caption term matching. If every content word of the query appears in a
   caption, those images are ground truth - they set both the count and the
   result set.
2. Score gradient. Where the similarity curve flattens, useful results have
   ended.

Measured over eight queries:

| Query                      | Basis          | Returned | Correct |
|----------------------------|----------------|----------|---------|
| "dog"                      | caption match  | 20       | 20      |
| "christmas tree"           | caption match  | 4        | 4       |
| "beach"                    | caption match  | 20       | 20      |
| "birthday cake"            | caption match  | 16       | 16      |
| "wedding"                  | caption match  | 11       | 11      |
| "swimming pool"            | caption match  | 6        | 6       |
| "people wearing sunglasses"| caption match  | 10       | 10      |
| "a red car"                | caption match  | 3        | 3       |
| "someone holding a baby"   | score gradient | 40       | partial |

The seven caption-matched queries returned 90 images with no false
positives. Compare the same corpus under fixed top-12: "dog" returned 10
correct of 12, missing 10 dogs that exist.

The two paths are not equivalent. Caption matching is exact. The gradient
path degrades: "someone holding a baby" found no literal match (the
stopword list does not cover "someone", and matching requires every content
word), fell through to the gradient, and hit its 40-result ceiling rather
than detecting a plateau. Results were relevant at the top and drifted to
adjacent concepts by rank 38 - babies present rather than babies held.

So the gradient path currently signals "no confident cutoff" by returning
its maximum, which reads like an answer but is not one.

## Note on SigLIP score scale

SigLIP is trained with a sigmoid loss and a learned temperature and bias, so
raw cosine similarities cluster near zero rather than in the 0.2-0.35 band
typical of CLIP. A score of 0.10 here is a strong match, not a weak one.

## Implications

- Deriving the result count is the single largest accuracy improvement made:
  100% precision on caption-matched queries against 83% on a fixed top-12,
  with double the recall.
- A bigger embedding model is not the fix for small-object retrieval - that
  was tested and regressed.
- Captions introduce a new failure mode: depicted-vs-present objects (the
  cat statue, a cat graphic on a shirt) match textually but are not what the
  user means. Qwen sometimes flags this unprompted, appending "no animals
  are visible besides the statue" - a signal that could be exploited.
- The gradient ceiling needs to report low confidence rather than return 40
  results.

## Not yet tested

- Whether the resolution gap holds on the post-2015 subset
- Hybrid fusion on queries where the two indexes disagree
- Partial caption matching (any content word rather than all) for queries
  like "someone holding a baby"
- Weighting fusion by query type rather than fusing equally
