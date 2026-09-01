"""Query understanding, without the model.

Everything here runs against a fixture roster and a hand-written model
reply. The model is the part that cannot be pinned - it is a 1.7B
instruct model and its answer to a given sentence is not a contract -
so what gets tested is what happens to that answer afterwards: what is
kept, what is dropped, and what the search ends up being asked.

The failure this file exists to prevent is a plausible wrong name
arriving at retrieval as a confident filter. A reader who searches for
sunglasses and silently gets one person's photographs has been given a
worse answer than a bad ranking: they have been given a confident one.
"""
import pytest

from visualmind import query

PEOPLE = {
    "Ada Fixture",
    "Bo Sample",
    "Cy Placeholder",
    "Robert One",
    "Robert Two",
}

EVENTS = {
    "event-001": {
        "id": "event-001",
        "name": "Fixture Picnic",
        "start": "2020-06-01",
        "end": "2020-06-01",
        "paths": {"img1.jpg", "img2.jpg"},
        "images": 2,
    },
    "event-002": {
        "id": "event-002",
        "name": "Sample Trip",
        "start": "2021-08-14",
        "end": "2021-08-15",
        "paths": {"img4.jpg"},
        "images": 1,
    },
}


def reply(monkeypatch, answer):
    """Stand in for the model with a fixed answer."""
    monkeypatch.setattr(query, "_generate", lambda *a, **k: answer)
    monkeypatch.setattr(query, "available", lambda: True)


def parse(monkeypatch, text, answer):
    reply(monkeypatch, answer)

    return query.parse(text, PEOPLE, EVENTS)


# ------------------------------------------------------- reading a reply


@pytest.mark.parametrize("answer,expected", [
    ('{"persons": [], "events": []}', {"persons": [], "events": []}),
    ('here you go: {"persons": []} thanks', {"persons": []}),
    ('```json\n{"persons": []}\n```', {"persons": []}),
])
def test_json_is_found_wherever_the_model_puts_it(answer, expected):
    """Small models wrap JSON in prose or a fence as often as not."""
    assert query.extract_json(answer) == expected


@pytest.mark.parametrize("answer", [
    "", None, "no json here", "{unclosed", "[1, 2]", "null", "{",
])
def test_an_unusable_reply_reads_as_nothing(answer):
    assert query.extract_json(answer) is None


def test_a_truncated_list_is_not_half_read():
    """The real failure: 30 events listed until the tokens ran out.

    A half-written object must not parse into a filter built from
    whichever names happened to fit.
    """
    answer = '{"persons": [], "events": ["Fixture Picnic", "Sample T'

    assert query.extract_json(answer) is None


# ------------------------------------------------------------ validation


def test_a_name_that_is_not_in_the_roster_is_dropped(monkeypatch):
    """The failure this layer exists to prevent."""
    out = parse(
        monkeypatch, "Zebediah at the beach",
        '{"persons": ["Zebediah Nonexistent"], "events": []}',
    )

    assert out["persons"] == []
    assert out["dropped"] == [
        {"text": "Zebediah Nonexistent", "as": "person", "why": "unknown"}
    ]


def test_an_ambiguous_name_is_dropped_rather_than_chosen(monkeypatch):
    """Two real people match. Picking one would be a guess."""
    out = parse(
        monkeypatch, "photos of Robert",
        '{"persons": ["Robert"], "events": []}',
    )

    assert out["persons"] == []
    assert out["dropped"][0]["why"] == "ambiguous"


def test_a_first_name_resolves_to_the_roster_spelling(monkeypatch):
    out = parse(
        monkeypatch, "Ada with sunglasses",
        '{"persons": ["Ada"], "events": []}',
    )

    assert out["persons"] == ["Ada Fixture"]


def test_a_name_the_query_never_said_is_dropped(monkeypatch):
    """Resolving is not enough; it has to have been asked for.

    "people wearing sunglasses" resolved to a roster entry called
    "_cartoon" against the real corpus - a real label, and a confident
    wrong filter.
    """
    out = parse(
        monkeypatch, "people wearing sunglasses",
        '{"persons": ["Cy Placeholder"], "events": []}',
    )

    assert out["persons"] == []
    assert out["dropped"][0]["why"] == "not named in the query"


def test_an_event_the_query_never_said_is_dropped(monkeypatch):
    out = parse(
        monkeypatch, "photos at the beach",
        '{"persons": [], "events": ["Fixture Picnic"]}',
    )

    assert out["events"] == []
    assert out["dropped"][0]["as"] == "event"


def test_an_event_the_query_named_is_kept(monkeypatch):
    out = parse(
        monkeypatch, "photos from the Fixture Picnic",
        '{"persons": [], "events": ["Fixture Picnic"]}',
    )

    assert out["events"] == ["event-001"]


def test_a_runaway_list_is_discarded_whole(monkeypatch):
    """Thirty events named "Happy Birthday ..." is not an answer.

    Which of them was meant is exactly what the model failed to decide,
    so none of them is chosen.
    """
    names = ", ".join('"Fixture Picnic"' for _ in range(30))
    out = parse(
        monkeypatch, "birthday cake",
        '{"persons": [], "events": [' + names + ']}',
    )

    assert out["events"] == []
    assert out["dropped"][0]["why"] == "too many to be a filter"
    assert out["dropped"][0]["text"] == "30 events"


def test_duplicates_resolve_once(monkeypatch):
    out = parse(
        monkeypatch, "Ada and Ada Fixture",
        '{"persons": ["Ada", "Ada Fixture"], "events": []}',
    )

    assert out["persons"] == ["Ada Fixture"]


# ----------------------------------------------------------- the terms


def test_a_name_is_taken_out_of_the_search_terms(monkeypatch):
    out = parse(
        monkeypatch, "Ada with sunglasses",
        '{"persons": ["Ada"], "events": []}',
    )

    assert out["terms"] == "with sunglasses"


def test_a_query_that_is_only_a_name_leaves_no_terms(monkeypatch):
    out = parse(
        monkeypatch, "Ada Fixture",
        '{"persons": ["Ada Fixture"], "events": []}',
    )

    assert out["terms"] == ""


def test_only_words_from_a_kept_name_are_removed(monkeypatch):
    """A dropped name must not take the query's words with it."""
    out = parse(
        monkeypatch, "Zebediah at the beach",
        '{"persons": ["Zebediah Nonexistent"], "events": []}',
    )

    assert out["terms"] == "Zebediah at the beach"


def test_a_stopword_inside_a_name_is_left_alone():
    """"A" in a name must not delete "a red car"."""
    assert query.remaining("a red car", ["A Person"], [], EVENTS) == \
        "a red car"


def test_punctuation_does_not_hide_a_name():
    assert query.remaining("Ada, at the beach", ["Ada Fixture"], [], EVENTS) \
        == "at the beach"


# ------------------------------------------------------------- fallback


def test_no_model_means_the_whole_query_is_terms(monkeypatch):
    monkeypatch.setattr(query, "available", lambda: False)

    out = query.parse("a red car", PEOPLE, EVENTS)

    assert out["terms"] == "a red car"
    assert out["persons"] == [] and out["events"] == []
    assert out["source"] == "fallback"


def test_a_model_that_fails_to_load_does_not_fail_the_query(monkeypatch):
    reply(monkeypatch, None)

    out = query.parse("a red car", PEOPLE, EVENTS)

    assert out["terms"] == "a red car"
    assert out["source"] == "fallback"
    assert "unavailable" in out["note"]


def test_an_unusable_reply_does_not_fail_the_query(monkeypatch):
    out = parse(monkeypatch, "a red car", "I cannot help with that.")

    assert out["terms"] == "a red car"
    assert out["source"] == "fallback"


def test_naming_nobody_falls_back_rather_than_claiming_a_parse(monkeypatch):
    """The answer is the same either way; the label should be honest."""
    out = parse(monkeypatch, "a red car", '{"persons": [], "events": []}')

    assert out["terms"] == "a red car"
    assert out["source"] == "fallback"


def test_an_empty_query_is_not_sent_to_the_model(monkeypatch):
    def explode(*a, **k):
        raise AssertionError("the model should not have been called")

    monkeypatch.setattr(query, "_generate", explode)

    assert query.parse("", PEOPLE, EVENTS)["terms"] == ""


@pytest.mark.parametrize("answer", [
    '{"persons": "Ada", "events": null}',
    '{"persons": [1, 2, 3], "events": {}}',
    '{"persons": [null], "events": [""]}',
    '{}',
])
def test_no_shape_of_reply_raises(monkeypatch, answer):
    """Whatever comes back, a query still runs."""
    out = parse(monkeypatch, "a red car", answer)

    assert out["terms"] in ("a red car", "")
    assert isinstance(out["persons"], list)


# ------------------------------------------------------- what it reports


def test_the_parse_reports_what_it_did(monkeypatch):
    """The reader has to be able to see a misread."""
    out = parse(
        monkeypatch, "Ada with sunglasses",
        '{"persons": ["Ada", "Nobody At All"], "events": []}',
    )

    assert out["query"] == "Ada with sunglasses"
    assert out["persons"] == ["Ada Fixture"]
    assert out["terms"] == "with sunglasses"
    assert out["dropped"] == [
        {"text": "Nobody At All", "as": "person", "why": "unknown"}
    ]
    assert out["source"] == "model"


def test_every_outcome_carries_the_same_keys(monkeypatch):
    """The API serialises this, so the shape cannot depend on the path."""
    keys = {"query", "persons", "events", "terms", "dropped",
            "source", "note"}

    got = parse(monkeypatch, "Ada", '{"persons": ["Ada"], "events": []}')
    fell_back = query.fallback("Ada", "because")

    assert set(got) == keys
    assert set(fell_back) == keys
