"""Answer types, ground-truth containers, and answer parsing.

The grading semantics live here, deliberately separate from both the task
templates and the reward rubric, so they can be unit-tested on hand-written
inputs with no database and no agent involved.

Tie handling
------------
Chinook is small and its prices take only two values, so ties are *pervasive*:
in 2021 eight different customers tie for 5th place by revenue. A "top 5" task
over that year has no single correct answer.

Rather than paper over that -- by rejecting those parameterisations, or by
bolting an unnatural "break ties by CustomerId ascending" clause onto the
question -- ranked ground truth is stored as an ordered sequence of **tie
groups**. Any linearisation that respects the group order is fully correct. So
if positions 5-12 all tie at $15.84, a top-5 answer may name any one of those
eight customers in position 5 and still score 1.0. This is the semantics an
analyst would actually accept, and it keeps the task space at full size.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum


class AnswerType(str, Enum):
    """How a submitted answer is compared to ground truth."""

    #: An ordered top-K list; position matters (subject to tie groups).
    RANKED_LIST = "ranked_list"
    #: An unordered set of entities selected by a predicate; order is ignored.
    ENTITY_SET = "entity_set"
    #: A single number, compared with tolerance.
    NUMERIC = "numeric"
    #: A single entity name (e.g. "which genre..."), possibly tied.
    CATEGORICAL = "categorical"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


# --------------------------------------------------------------------------
# normalisation
# --------------------------------------------------------------------------

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_SPACE = re.compile(r"\s+")


def normalize_entity(value: object) -> str:
    """Casefold, strip accents and punctuation, collapse whitespace.

    Makes ``"Antônio Carlos Jobim"``, ``"antonio carlos jobim"`` and
    ``"Antonio  Carlos Jobim."`` compare equal, without being loose enough to
    conflate two genuinely different names.
    """
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = _PUNCT.sub(" ", text)
    return _SPACE.sub(" ", text).strip().casefold()


@dataclass(frozen=True)
class Entity:
    """One answerable item, plus the spellings that count as naming it.

    ``aliases`` typically holds the normalised display name and the numeric
    primary key. Accepting the key is a deliberate leniency -- an agent that
    reports ``CustomerId 42`` has demonstrably done the work -- and it is not
    exploitable, because keys are exact identifiers, not fuzzy matches.
    """

    key: str
    display: str
    aliases: frozenset[str] = field(default_factory=frozenset)

    @staticmethod
    def make(key: object, display: str, *extra: object) -> Entity:
        aliases = {normalize_entity(display), normalize_entity(key)}
        aliases.update(normalize_entity(e) for e in extra)
        aliases.discard("")
        return Entity(key=str(key), display=display, aliases=frozenset(aliases))

    def matches(self, candidate: str) -> bool:
        return normalize_entity(candidate) in self.aliases


@dataclass(frozen=True)
class GroundTruth:
    """The correct answer, computed in pandas independently of any SQL.

    Only the fields relevant to ``answer_type`` are populated.
    """

    answer_type: AnswerType
    #: RANKED_LIST / CATEGORICAL: ordered tie groups. Each inner tuple holds
    #: entities that share a value and are therefore interchangeable.
    tie_groups: tuple[tuple[Entity, ...], ...] = ()
    #: RANKED_LIST: how many positions the answer must fill.
    k: int = 0
    #: ENTITY_SET: the full, unordered correct set.
    entities: tuple[Entity, ...] = ()
    #: NUMERIC: the correct value and its comparison tolerances.
    value: float | None = None
    abs_tol: float = 0.01
    rel_tol: float = 1e-3
    #: CATEGORICAL: the other candidates in the same universe (e.g. every other
    #: genre). Used to make the prose fallback in ``_score_categorical`` safe:
    #: an answer that mentions several candidates is ambiguous, not correct.
    distractors: tuple[Entity, ...] = ()
    #: Human-readable rendering, for eval logs and debugging.
    notes: str = ""

    def describe(self) -> str:
        if self.answer_type is AnswerType.NUMERIC:
            return f"{self.value}"
        if self.answer_type is AnswerType.ENTITY_SET:
            return "{" + ", ".join(sorted(e.display for e in self.entities)) + "}"
        parts = []
        position = 1
        for group in self.tie_groups:
            names = " | ".join(e.display for e in group)
            if len(group) == 1:
                parts.append(f"{position}. {names}")
            else:
                # A tie group occupies a *band* of ranks; any ordering of its
                # members within that band is correct.
                parts.append(f"{position}-{position + len(group) - 1}. {names} (tied)")
            position += len(group)
        return "; ".join(parts)

    def flat_entities(self) -> tuple[Entity, ...]:
        if self.answer_type is AnswerType.ENTITY_SET:
            return self.entities
        return tuple(e for group in self.tie_groups for e in group)


# --------------------------------------------------------------------------
# parsing submitted answers
# --------------------------------------------------------------------------

_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_LIST_ITEM_PREFIX = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s*")
_TRAILING_ANNOTATION = re.compile(r"\s*[(\[].*?[)\]]\s*$")


def parse_numeric(answer: str) -> float | None:
    """Pull a single number out of a free-text answer.

    Tolerates ``"$1,234.56"``, ``"The total was 1234.56 dollars."`` and bare
    ``"1234.56"``. Returns ``None`` when the text contains no number, or when
    it contains several different ones (ambiguous -- the agent was asked for
    one figure and did not give one).
    """
    if answer is None:
        return None
    text = str(answer).strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
    except (ValueError, TypeError):
        pass
    else:
        if isinstance(payload, (int, float)) and not isinstance(payload, bool):
            return float(payload)
        if isinstance(payload, dict) and len(payload) == 1:
            only = next(iter(payload.values()))
            if isinstance(only, (int, float)) and not isinstance(only, bool):
                return float(only)

    candidates = {float(m.group().replace(",", "")) for m in _NUMBER.finditer(text)}
    if len(candidates) != 1:
        return None
    return candidates.pop()


def parse_entity_list(answer: str) -> list[str]:
    """Split a free-text answer into an ordered list of entity names.

    Accepts a JSON array, a newline- or comma-separated list, and numbered or
    bulleted lists. Strips trailing parenthetical annotations such as
    ``"Helena Holy ($49.62)"`` so that a correct name is not rejected for
    carrying its revenue figure alongside.
    """
    if answer is None:
        return []
    text = str(answer).strip()
    if not text:
        return []

    try:
        payload = json.loads(text)
    except (ValueError, TypeError):
        payload = None
    if isinstance(payload, list):
        return [_clean_item(str(item)) for item in payload if str(item).strip()]
    if isinstance(payload, dict):
        for value in payload.values():
            if isinstance(value, list):
                return [_clean_item(str(i)) for i in value if str(i).strip()]

    lines = [ln for ln in (l.strip() for l in text.splitlines()) if ln]
    # A multi-line answer is a list of lines; a single line is comma-separated.
    items = lines if len(lines) > 1 else re.split(r",(?![^(]*\))", text)
    return [cleaned for cleaned in (_clean_item(i) for i in items) if cleaned]


def _clean_item(item: str) -> str:
    item = _LIST_ITEM_PREFIX.sub("", item.strip())
    item = _TRAILING_ANNOTATION.sub("", item)
    # Drop a trailing "- $12.34" / ": 12.34" style annotation too.
    item = re.sub(r"\s*[-:–—]\s*\$?-?[\d,]+(?:\.\d+)?\s*$", "", item)
    return item.strip().strip("\"'").strip()


# --------------------------------------------------------------------------
# grading
# --------------------------------------------------------------------------


def _match_index(candidate: str, entities: tuple[Entity, ...]) -> int | None:
    for index, entity in enumerate(entities):
        if entity.matches(candidate):
            return index
    return None


def score_answer(answer: str, truth: GroundTruth) -> float:
    """Score a submitted answer against ground truth, in ``[0, 1]``.

    Pure function of ``(answer, truth)`` -- no database, no trajectory, no
    model. Partial credit is granted so that the training signal is not a
    cliff, but only for genuinely partial answers.
    """
    if truth.answer_type is AnswerType.NUMERIC:
        return _score_numeric(answer, truth)
    if truth.answer_type is AnswerType.ENTITY_SET:
        return _score_set(answer, truth)
    if truth.answer_type is AnswerType.CATEGORICAL:
        return _score_categorical(answer, truth)
    return _score_ranked(answer, truth)


def _score_numeric(answer: str, truth: GroundTruth) -> float:
    submitted = parse_numeric(answer)
    if submitted is None or truth.value is None:
        return 0.0
    delta = abs(submitted - truth.value)
    if delta <= truth.abs_tol:
        return 1.0
    if truth.value != 0 and delta / abs(truth.value) <= truth.rel_tol:
        return 1.0
    return 0.0


def _score_set(answer: str, truth: GroundTruth) -> float:
    """Jaccard similarity between the submitted and correct sets.

    Jaccard, not recall, so that dumping every entity in the catalogue -- which
    would trivially achieve recall 1.0 -- scores near zero instead.
    """
    submitted = parse_entity_list(answer)
    if not truth.entities:
        return 1.0 if not submitted else 0.0
    matched: set[int] = set()
    unmatched = 0
    seen: set[str] = set()
    for item in submitted:
        norm = normalize_entity(item)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        index = _match_index(item, truth.entities)
        if index is None:
            unmatched += 1
        else:
            matched.add(index)
    union = len(truth.entities) + unmatched
    return len(matched) / union if union else 0.0


def _score_categorical(answer: str, truth: GroundTruth) -> float:
    acceptable = truth.tie_groups[0] if truth.tie_groups else ()
    if not acceptable:
        return 0.0

    items = [item for item in parse_entity_list(answer) if item]
    if not items:
        return 0.0

    # Exact path: every named option must be a correct (or tied-correct) one,
    # and naming more options than there are ties is hedging, not answering.
    if len(items) <= len(acceptable) and all(
        _match_index(item, acceptable) is not None for item in items
    ):
        return 1.0

    # Prose fallback: "The top genre in Brazil is Rock." parses as one long
    # item that matches nothing. Accept it only when exactly one candidate from
    # the whole universe is mentioned, so listing several cannot score.
    haystack = f" {normalize_entity(answer)} "
    mentioned = {
        entity.key
        for entity in (*acceptable, *truth.distractors)
        for alias in entity.aliases
        if alias and f" {alias} " in haystack
    }
    if len(mentioned) == 1:
        return 1.0 if mentioned <= {e.key for e in acceptable} else 0.0
    return 0.0


def _score_ranked(answer: str, truth: GroundTruth) -> float:
    """Fraction of the K positions filled by an entity from the right tie group.

    Duplicates are not rewarded twice: each ground-truth entity may satisfy at
    most one position.
    """
    k = truth.k or sum(len(g) for g in truth.tie_groups)
    if k == 0:
        return 0.0

    # Expand tie groups into the rank band each entity may legitimately occupy.
    band_of: dict[str, tuple[int, int]] = {}
    entity_by_key: dict[str, Entity] = {}
    position = 0
    for group in truth.tie_groups:
        lo, hi = position, position + len(group) - 1
        for entity in group:
            band_of[entity.key] = (lo, hi)
            entity_by_key[entity.key] = entity
        position += len(group)

    submitted = parse_entity_list(answer)[:k]
    used: set[str] = set()
    correct = 0
    for index, item in enumerate(submitted):
        for key, entity in entity_by_key.items():
            if key in used or not entity.matches(item):
                continue
            lo, hi = band_of[key]
            if lo <= index <= hi:
                correct += 1
                used.add(key)
            break
    return correct / k
