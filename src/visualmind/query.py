"""Turn a raw query into structured intent, before retrieval sees it.

A layer above retrieval, not a change to it. retrieval.search still takes
persons, event names and query text exactly as it did; this decides what
those three should be when all the caller has is a sentence.

The model runs locally. A query carries family names, and this project
does not send photographs or the words used to find them off the
machine, so the extraction happens on the same card that holds the
encoders.

Two things are load-bearing.

Nothing the model says about identity is trusted. Every name it proposes
is resolved against the real roster with people.resolve and
events.resolve - the same functions a typed --person goes through - and
anything that does not resolve, or resolves ambiguously, is dropped
rather than guessed at. A 1.7B model asked for names will happily
produce a plausible one that nobody in the corpus is called, and a
plausible wrong name arriving as a confident filter is worse than no
filter at all: the reader gets a confident, empty, wrongly-labelled
answer instead of a search.

And the parse is reported, not just applied. What became a person, what
became search terms, and what was thrown away all travel back with the
result, because a reader who types "Bob with sunglasses" needs to be
able to see that it was read as a person plus a term rather than as
three words about sunglasses. It is the same obligation the basis line
already meets when it admits that results came from a gradient.

When the model is missing, fails to load, or returns something
unusable, the whole query becomes search terms and the search runs
anyway. A query must never fail because a model did not load, so every
path through here has a fallback and none of them raise.
"""
import json
import re
from pathlib import Path

import yaml

from visualmind import events as events_module
from visualmind import people, retrieval

MODEL_CONFIG = Path("configs/models.yaml")
ROLE = "query_understanding"

# Long enough for a JSON object naming a few people; short enough that a
# model which starts writing an essay is cut off rather than indulged.
MAX_NEW_TOKENS = 160

# Thinking mode is off. Qwen3 emits a <think> block by default, which is
# both slower and another thing to strip before the JSON is reachable.
ENABLE_THINKING = False

PROMPT = """You identify people and events named in a photo-library query.

Known people:
{people}

Known events:
{events}

Return ONLY a JSON object, nothing before or after it, with exactly:
  "persons": people the query names, from the list above
  "events":  events the query names, from the list above

Rules:
- Use names exactly as they appear above. Never invent one.
- A first name on its own counts. If the query says a first name that
  belongs to exactly one person above, return that person's full name.
- A descriptive word is not an event just because event names contain
  it. "birthday cake" is a thing to look for, not an occasion to filter
  by. Only fill "events" when the query names one specific occasion.
- Many queries name nobody and no event. Empty lists are a normal answer.

Examples:
Query: dog
JSON: {{"persons": [], "events": []}}
Query: birthday cake
JSON: {{"persons": [], "events": []}}
Query: {example_first} with sunglasses
JSON: {{"persons": ["{example_name}"], "events": []}}

Query: {query}
JSON:"""

# A query naming more than a couple of people or occasions is not a
# query, it is the model matching a common word against every roster
# entry containing it: "birthday cake" once produced thirty events named
# "Happy Birthday ..." and ran out of tokens before closing the JSON.
# A runaway list is discarded whole rather than filtered down, because
# which of the thirty it meant is exactly what it failed to decide.
MAX_FILTERS = 3

_HELD = {}


def available():
    """Whether a model is registered for this role."""
    return bool(_config())


def _config():
    if not MODEL_CONFIG.exists():
        return None

    try:
        config = yaml.safe_load(MODEL_CONFIG.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        return None

    entry = (config or {}).get("models", {}).get(ROLE)

    if not entry or not entry.get("repo_id"):
        return None

    return entry


def release():
    """Drop the held model. Test seam, and a way to reclaim the card."""
    _HELD.clear()


def _load():
    """The tokenizer and model, held after the first call.

    Returns None rather than raising when anything at all goes wrong -
    no config, no weights, no CUDA, a transformers version that does not
    know this architecture. Every one of those is a reason to fall back
    to a plain term search, not a reason to fail a query.
    """
    if "model" in _HELD:
        return _HELD["model"]

    entry = _config()

    if entry is None:
        return None

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            entry["repo_id"], revision=entry.get("revision")
        )
        model = AutoModelForCausalLM.from_pretrained(
            entry["repo_id"],
            revision=entry.get("revision"),
            dtype=torch.bfloat16,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
        ).eval()

        _HELD["model"] = (tokenizer, model, torch)
    except Exception:
        # Held as a sentinel so a broken load is attempted once per
        # process rather than on every query.
        _HELD["model"] = None

    return _HELD["model"]


def _generate(text, people_names, event_names):
    """The model's raw answer, or None if it could not produce one."""
    loaded = _load()

    if loaded is None:
        return None

    tokenizer, model, torch = loaded

    prompt = PROMPT.format(
        people="\n".join("- " + name for name in people_names) or "- (none)",
        events="\n".join("- " + name for name in event_names) or "- (none)",
        # The worked example uses a real roster name. With an invented
        # one the model copied it into its answer for unrelated queries,
        # which validation then dropped - a filter lost to a name that
        # only ever existed in the prompt.
        example_name=people_names[0] if people_names else "Ada Fixture",
        example_first=(people_names[0] if people_names
                       else "Ada Fixture").split()[0],
        query=text,
    )

    try:
        messages = [{"role": "user", "content": prompt}]
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=ENABLE_THINKING,
        )
        inputs = tokenizer([rendered], return_tensors="pt").to(model.device)

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )

        return tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
    except Exception:
        return None


OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(answer):
    """The first JSON object in a model's answer, or None.

    Small instruct models wrap JSON in prose or a code fence about as
    often as they do not, so the object is located rather than assumed
    to be the whole reply.
    """
    if not answer:
        return None

    found = OBJECT.search(answer)

    if not found:
        return None

    try:
        parsed = json.loads(found.group(0))
    except (json.JSONDecodeError, ValueError):
        return None

    return parsed if isinstance(parsed, dict) else None


def _strings(value):
    """A list of non-empty strings from whatever the model returned."""
    if isinstance(value, str):
        value = [value]

    if not isinstance(value, list):
        return []

    return [item.strip() for item in value
            if isinstance(item, str) and item.strip()]


def words_of(text):
    """Comparable words: alphanumeric, lowercased, stopwords removed."""
    found = {part.lower() for part in re.findall(r"[^\W_]+", text or "")}

    return {word for word in found - retrieval.STOPWORDS if len(word) > 1}


def mentioned(name, text):
    """Whether the query actually contains a word from this name.

    A name can resolve against the roster and still be one the reader
    never typed. "people wearing sunglasses" resolved to a roster entry
    called "_cartoon" - a real label, and a confident wrong filter,
    since the query says nothing about cartoons. Resolution proves the
    name exists; this proves it was asked for, and both are needed
    before a filter is put on somebody's photographs.
    """
    return bool(words_of(name) & words_of(text))


def validate_people(proposed, known, text=""):
    """Resolve proposed names against the roster. Drop what does not.

    people.resolve is the same function a typed --person goes through,
    so a name the model proposes is held to exactly the standard a name
    the reader typed is held to. Unknown and ambiguous both drop: the
    first because nobody is called that, the second because choosing
    between two real people is a guess, and this is the layer that is
    not allowed to guess about identity.
    """
    kept = []
    dropped = []

    for name in proposed:
        try:
            resolved = people.resolve(name, known)
        except people.UnknownName:
            dropped.append({"text": name, "as": "person", "why": "unknown"})
            continue
        except people.AmbiguousName:
            dropped.append({"text": name, "as": "person", "why": "ambiguous"})
            continue

        if text and not mentioned(resolved, text):
            dropped.append({"text": resolved, "as": "person",
                            "why": "not named in the query"})
            continue

        if resolved not in kept:
            kept.append(resolved)

    return kept, dropped


def validate_events(proposed, known, text=""):
    """Resolve proposed event references. Drop what does not resolve."""
    kept = []
    dropped = []

    for reference in proposed:
        try:
            resolved = events_module.resolve(reference, known)
        except events_module.UnknownEvent:
            dropped.append({"text": reference, "as": "event",
                            "why": "unknown"})
            continue
        except events_module.AmbiguousEvent:
            dropped.append({"text": reference, "as": "event",
                            "why": "ambiguous"})
            continue

        entry = known.get(resolved) or {}

        if text and not mentioned(
            (entry.get("name") or "") + " " + resolved, text
        ):
            dropped.append({"text": entry.get("name") or resolved,
                            "as": "event", "why": "not named in the query"})
            continue

        if resolved not in kept:
            kept.append(resolved)

    return kept, dropped


def _bounded(proposed, kind):
    """Discard a list that is too long to be an answer.

    Reported as one entry rather than as thirty, because the failure is
    the length of the list and naming every member of it would bury the
    rest of the parse.
    """
    if len(proposed) <= MAX_FILTERS:
        return proposed, []

    return [], [{
        "text": str(len(proposed)) + " " + kind + "s",
        "as": kind,
        "why": "too many to be a filter",
    }]


def remaining(text, persons, events, events_known):
    """The query with the identified names taken out of it.

    Computed here rather than asked of the model. Given the job of
    rewriting the query as well as reading it, a 1.7B model returned the
    name it had just extracted still sitting in the search terms, and
    trimmed "people wearing sunglasses" to "sunglasses" - it is good at
    spotting which roster entry a word refers to and unreliable at
    editing a sentence. Removal is mechanical, so it is done mechanically
    and the model is only asked the part it is good at.

    Only words that belong to a name that actually resolved are removed,
    and never a stopword: "May" is a month as well as potentially a
    person, and a query is more often about the month.
    """
    taken = set()

    for name in persons:
        taken.update(part.lower() for part in re.findall(r"[^\W_]+", name))

    for event_id in events:
        entry = events_known.get(event_id) or {}
        taken.update(
            part.lower()
            for part in re.findall(r"[^\W_]+", entry.get("name", ""))
        )
        taken.add(event_id.lower())

    taken -= retrieval.STOPWORDS
    taken = {word for word in taken if len(word) > 1}

    kept = [
        word for word in text.split()
        if re.sub(r"[^\w]", "", word).lower() not in taken
    ]

    return " ".join(kept).strip()


def fallback(text, note):
    """Today's behaviour: the whole query is search terms."""
    return {
        "query": text,
        "persons": [],
        "events": [],
        "terms": text,
        "dropped": [],
        "source": "fallback",
        "note": note,
    }


def parse(text, people_known=None, events_known=None):
    """Structured intent for a raw query.

    `people_known` is the roster of names, `events_known` the event index
    both as people.index() and events.index() already return them. Both
    default to reading the real ones.

    Never raises. The worst case is the fallback, which is what the
    search did before this layer existed.
    """
    text = (text or "").strip()

    if not text:
        return fallback(text, "no query text")

    if people_known is None:
        images, _ = people.index()
        people_known = set(images)

    if events_known is None:
        events_known = events_module.index()

    if not available():
        return fallback(text, "no model registered for " + ROLE)

    event_names = [
        entry["name"] for entry in events_known.values()
        if entry.get("name")
    ]

    answer = _generate(text, sorted(people_known), sorted(set(event_names)))

    if answer is None:
        return fallback(text, "model unavailable")

    parsed = extract_json(answer)

    if parsed is None:
        return fallback(text, "model returned no usable JSON")

    return interpret(parsed, text, people_known, events_known)


def interpret(parsed, text, people_known, events_known):
    """Validate a parsed object into intent. The pure half of parse().

    Separate from the model call so the whole of this can be tested
    against a fixture roster without loading anything.
    """
    proposed_people, runaway_people = _bounded(
        _strings(parsed.get("persons")), "person"
    )
    proposed_events, runaway_events = _bounded(
        _strings(parsed.get("events")), "event"
    )

    persons, dropped_people = validate_people(
        proposed_people, people_known, text
    )
    events, dropped_events = validate_events(
        proposed_events, events_known, text
    )

    dropped_people = runaway_people + dropped_people
    dropped_events = runaway_events + dropped_events

    dropped = dropped_people + dropped_events
    terms = remaining(text, persons, events, events_known)

    if not persons and not events:
        # Nothing proposed at all is the answer the fallback gives, and
        # calling it a parse would claim an understanding that did not
        # happen.
        if not dropped:
            return fallback(text, "no person or event named")

        # But everything proposed being rejected is a different fact,
        # and the one most worth seeing: the model did read a name into
        # this query and it was not allowed through. Falling back here
        # would run the same search while throwing away the only record
        # that a name was ever considered.
        return {
            "query": text,
            "persons": [],
            "events": [],
            "terms": terms,
            "dropped": dropped,
            "source": "model",
            "note": "nothing proposed survived validation",
        }

    return {
        "query": text,
        "persons": persons,
        "events": events,
        "terms": terms,
        "dropped": dropped,
        "source": "model",
        "note": "",
    }
