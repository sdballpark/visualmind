# Duplicate detection evaluation

Three-tier duplicate detection over the same 441-image corpus. Each tier
catches a different kind of redundancy, and each is cheap enough to run
before the next.

    EXACT    identical SHA-256
    NEAR     perceptual hash within Hamming distance 6
    SIMILAR  DINOv2 cosine similarity >= 0.92

The script reports only. It never moves, renames, or deletes.

## Results

| Tier    | Groups | Notes                                    |
|---------|--------|------------------------------------------|
| EXACT   | 0      | upstream ingestion already deduplicated  |
| NEAR    | 6      | 6 redundant files                        |
| SIMILAR | 15     | 13 pairs, 2 triples                      |

## Finding 1 - zero exact duplicates confirms the ingestion pipeline

The Gmail downloader that produced this archive hashes attachments and
skips byte-identical repeats. Finding no EXACT groups is a positive result:
it independently confirms that upstream deduplication works, using a
different code path than the one that performed it.

## Finding 2 - NEAR catches what SHA-256 structurally cannot

All six NEAR groups are the same photograph in different encodings:

| Kind                    | Example                                  |
|-------------------------|------------------------------------------|
| Re-sent, re-encoded     | image.png / image.png, both 485x353      |
| Renamed                 | image_3.png / DSC01235.JPG, both 2592x1944 |
| Half-scale copy         | image_2.png 3264x1592 / LisaJer5.jpg 1632x796 |
| Suffixed duplicate      | 20160520_190326.jpg / _2.jpg, both 5312x2988 |

These are invisible to hashing - one re-encode changes every byte - and
uninteresting to a semantic index, which would rank them as merely similar.
Perceptual hashing is the right tool and it found all of them cheaply.

The keeper heuristic (most pixels, then largest file, then earliest EXIF)
selected the higher-resolution copy in every case.

## Finding 3 - SIMILAR groups sort into four kinds

Reviewing all 15 groups against their captions:

**Same photograph, different crop (6 groups).** A file paired with itself
at different dimensions. The captions differ in a revealing way: one group
reads "Five children posing outdoors at night" against "Four children
posing indoors" - the crop removed a person and changed the apparent
setting. Another reads "Five women seated" against "Four women seated".
The caption is describing the crop, not the original.

**Same scene, different frame (3 groups).** Genuine burst captures. One
group holds three TomKathParty frames captioned "ten men", "nine men", and
"nine men" - the same gathering seconds apart.

**Same event, different subject (4 groups).** A rocky overlook, a family on
rocks, a group on a lawn. Correct at the event level rather than the photo
level, which is useful for album grouping but is not duplication.

**Text cards (2 groups).** Birthday message graphics. One group pairs
"HAPPY BIRTHDAY AL..." with "I LOVE, UNCLE BOB IN VEGAS" - different text,
but DINOv2 sees black background with coloured lettering. A visual-
similarity true positive and a semantic false positive.

Roughly 13 of 15 groups are useful for the intended purpose. The two text-
card groups are correct about what they measure and wrong about what a user
would mean.

## Finding 4 - the crop cases expose a captioning assumption

Six SIMILAR groups pair a file with a cropped version of itself, and the
captions disagree about how many people are present. This is not a
captioning error - Qwen described each image accurately.

It does mean caption-derived counts are per-file, not per-photograph. A
query for "five children" would match the uncropped file and miss the crop.
Deduplicating before captioning would avoid this, at the cost of losing the
crop as a separate searchable item.

## Thresholds

Both thresholds were set by judgement, not tuning:

- pHash Hamming distance 6 of 64 bits. Standard practice for
  near-duplicate work; produced no false positives here.
- DINOv2 cosine 0.92. Produced 15 groups with roughly 87% usefulness.
  Lowering it would catch more same-event pairs and more text cards.

Neither has been swept. With 441 images the cost of a mistake is low
enough that manual review is practical.

## Not yet tested

- Threshold sweeps against a labelled duplicate set
- Whether pHash distance 8-10 finds additional real duplicates
- Whether excluding text-card images (detectable from captions) before the
  SIMILAR pass removes both false-positive groups
- Best-shot selection within SIMILAR groups: sharpness, exposure, eyes open
