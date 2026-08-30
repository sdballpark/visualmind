# Event grouping evaluation

Grouping the 441-image corpus into occasions - a weekend, a birthday, a
holiday - using metadata already present rather than a new model.

## The date problem

Three dates exist per image and they disagree:

| Source              | Coverage | What it means            |
|---------------------|----------|--------------------------|
| EXIF capture time   | 288      | when the photo was taken |
| Gmail message date  | 400      | when it was emailed      |
| Gmail folder year   | 400      | the same, coarser        |

The gap between them is not small. A baptism photo with EXIF 2006 arrived
in a 2026 message. Grouping by message date would produce "photos Terry
emailed in February 2026" - a thread, not an occasion.

## Finding 1 - email threads are retrospective, not contemporaneous

The obvious shortcut was to treat an email thread as an event. Someone
attached those photos to that message deliberately, which is human
curation for free.

It does not hold:

    threads with 2+ dated images:      66
    threads spanning multiple years:   28

"Re: Happy Birthday Daniel" contains photos from 2000 through 2023.
Birthday threads accumulate old pictures as people reply with memories.

Threads were therefore demoted from primary signal to fallback.

## Finding 2 - EXIF needs validating, not just parsing

Two failure modes appear in this corpus:

- The zero date. Cameras emit `0000:00:00 00:00:00` when the clock was
  never set. Parsed naively this sorts before every real photograph and
  creates a spurious "year 0" event.
- Out-of-range dates from clock resets.

`parse_exif` rejects both: the zero prefix outright, and anything outside
1990 to the present year. 288 images survive validation, slightly fewer
than the 292 with a non-empty `best_exif_date` field.

## Finding 3 - a 72-hour gap suits an occasion-based corpus

Photos are sorted by capture time; a gap longer than the threshold starts
a new event.

The threshold encodes what counts as one occasion:

| Gap  | Effect                                                    |
|------|-----------------------------------------------------------|
| 6h   | an afternoon party and an evening dinner split            |
| 24h  | one calendar day is one event; a late wedding splits      |
| 72h  | a weekend trip or multi-day holiday stays whole           |

72 hours was chosen. This corpus is occasions rather than continuous
documentation, and the largest resulting event - 32 photos across
2005-09-17 and 18 - was confirmed by inspection to be a single weekend.

Result: 83 events from 288 dated photographs.

## Finding 4 - undated photos should stay unplaced

Photos without EXIF fall back to their thread. If every dated photo in
that thread landed in one event, undated siblings join it. 35 images
placed this way.

118 remain unassigned. Two approaches to reducing that were tested and
both rejected.

**Visual similarity.** If an undated photo closely resembles a dated one,
it plausibly belongs to the same occasion. Measured against the existing
DINOv2 index:

    unassigned images with a >=0.85 match to a dated photo:  5 of 118

The signal is not there. Visual similarity finds duplicates, not
co-occurrence: two photos from the same party do not resemble each other
more than two photos of any family gathering.

**Dominant-event voting.** If a thread maps to several events but one
holds a clear majority of its dated photos, place undated siblings there.

    would place with a 60% dominant event:  11 of 118

Not worth the added complexity or the risk of a confidently wrong date.

Inspecting the 118 shows why they resist placement. Many are text cards
and birthday graphics with no capture time and nothing to place them
against. Others are scans of prints. Only 20 have no email thread at all;
the remaining 98 have threads that genuinely span multiple occasions.

Leaving them unassigned is the honest outcome, and it is visible in the
output rather than hidden.

## Finding 5 - names come free where threads agree

An event borrows its email subject when at least 60% of its images share
one. 17 of 83 events are named this way - "Happy Birthday Andrew (Nov
2004)", "Happy Birthday Leilani (May 2016)".

The rest carry a date label. That is the weak point: Christmas Day 2023
holds 18 images and is called "Dec 2023", because no email subject said
Christmas. The captions do mention Christmas trees, so caption-derived or
calendar-derived naming would close most of the gap. Not yet built.

## Composition with person filtering

Events and people compose as filters. Events union with each other, since
an image belongs to exactly one event; people intersect, since a photo can
contain several.

    --event 2005-09                    32 images
    --person lisa                      73 images
    --event 2005-09 --person lisa       2 images

## Not yet tested

- Whether a 24-hour gap would split any event that should stay whole
- Calendar-based naming for Christmas, Thanksgiving, New Year
- Caption-derived naming where email subjects are silent
- Whether face co-occurrence could place undated photos where visual
  similarity failed
