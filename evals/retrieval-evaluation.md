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
return from three signals, in descending order of confidence:

1. Full caption term match. Every content word of the query appears in a
   caption. Those images set both the count and the result set.
2. Partial caption term match. At least two content words appear.
3. Score gradient. Where the similarity curve flattens, useful results have
   ended.

Measured over nine queries:

| Query                       | Basis         | Returned | Correct |
|-----------------------------|---------------|----------|---------|
| "dog"                       | full match    | 20       | 20      |
| "christmas tree"            | full match    | 4        | 4       |
| "beach"                     | full match    | 20       | 20      |
| "birthday cake"             | full match    | 16       | 16      |
| "wedding"                   | full match    | 11       | 11      |
| "swimming pool"             | full match    | 6        | 6       |
| "people wearing sunglasses" | full match    | 30       | 28      |
| "a red car"                 | full match    | 3        | 3       |
| "someone holding a baby"    | full match    | 18       | 15      |

Compare the same corpus under fixed top-12: "dog" returned 10 correct of 12,
missing 10 dogs that exist.

### Precision splits by query type

| Query type      | Queries | Images | Precision |
|-----------------|---------|--------|-----------|
| Concrete nouns  | 8       | 110    | 98%       |
| Relational      | 1       | 18     | 83%       |

Term matching is exact when the query names objects. It degrades when the
query expresses a relation between terms, because bag-of-words matching has
no notion of subject and object.

The three misses on "someone holding a baby" are all of that kind: two
photos of a baby holding something else, and a baby shower invitation where
a person holds a card. Every one contains both "holding" and "baby" in its
caption; none shows a person holding a baby.

### Stopwords matter more than expected

"someone holding a baby" initially found no full match and fell through to
the gradient path, returning 40 results. The only blocker was the word
"someone", which appears in no caption. Adding it and similar generic terms
("people", "photo", "person") to the stopword list converted the query from
a 40-result guess to an 18-result set at 83% precision.

### That stopword change moved a second query, unmeasured

The "people wearing sunglasses" row above read 10 returned, 10 correct
until this revision. The query returns 30, hand-labelled as 28 correct.
The two exceptions are a man with sunglasses around his neck who is
wearing reading glasses, counted as a miss, and a photograph of children
ice skating.

The term-matching fix did not cause this. That change was captured
before and after against all ten queries cited in this document and left
every one byte-identical in count, membership, and ordering; the count
was already 30 beforehand.

It moved with the stopword change described just above. Adding "people",
"person" and "photo" to the list to rescue "someone holding a baby" also
dropped "people" from this query's required terms, turning a three-term
match into a two-term one on "wearing" and "sunglasses". The stricter
three-term match had been hiding 18 correct images. Recall tripled;
precision for the query fell from 100% to 93%, which moves the
concrete-noun split above from 100% to 98%.

The process finding matters more than the row. A change made for one
query silently moved another query's published number, and nobody re-ran
the affected query afterwards, so the figure stayed here, wrong, until a
baseline capture for an unrelated change surfaced it. Catching exactly
that is what this document is for. The stopword list, the tokeniser and
the term-match rule are shared by every query, so the nine above are a
suite to re-run whenever one of them changes, not a set of results to
quote.

## Finding 7 - semantic reranking did not fix relational queries

Finding 6 suggested weighting the caption semantic score within the
term-matched set, on the theory that BGE understands "holding a baby" as a
relation even though term matching does not. Tested and rejected.

Reordering the 18-image set by caption score instead of RRF left precision
unchanged at 15/18. The two subject/object errors ranked 14th and 15th of
18 - mid-pack, with two correct matches below them. BGE does not separate
these cases.

Adding a trim, discarding entries whose caption score trails the set:

| Approach                  | Returned | Correct | Precision |
|---------------------------|----------|---------|-----------|
| RRF order, no trim        | 18       | 15      | 83.3%     |
| Caption order, no trim    | 18       | 15      | 83.3%     |
| Caption order, with trim  | 16       | 14      | 87.5%     |

The trim bought 4 points of precision by discarding one true match and one
false one. Since the mechanism it relied on does not hold, that gain is
incidental.

Trimming ships behind a `--trim` flag, off by default. A false positive is
visible and can be ignored; a silently omitted true match cannot. For a
personal archive that asymmetry decides it.

Caption-score ordering is kept as the default within matched sets - no worse
than RRF and easier to explain.

## Note on SigLIP score scale

SigLIP is trained with a sigmoid loss and a learned temperature and bias, so
raw cosine similarities cluster near zero rather than in the 0.2-0.35 band
typical of CLIP. A score of 0.10 here is a strong match, not a weak one.

## Implications

- Deriving the result count is the single largest accuracy improvement made:
  100% precision on noun queries against 83% on a fixed top-12, with double
  the recall.
- A bigger embedding model is not the fix for small-object retrieval - that
  was tested and regressed.
- Neither is semantic reranking the fix for relational queries - also tested
  and regressed.
- Captions introduce a new failure mode: depicted-vs-present objects (the
  cat statue, a cat graphic on a shirt) match textually but are not what the
  user means. Qwen sometimes flags this unprompted, appending "no animals
  are visible besides the statue" - a signal that could be exploited.

## Not yet tested

- Whether the resolution gap holds on the post-2015 subset
- Hybrid fusion on queries where the two indexes disagree
- Whether a caption prompt that names subject-object relations explicitly
  would fix the relational failures at the source
- Whether an LLM pass over the matched captions could filter subject/object
  errors that embeddings cannot
