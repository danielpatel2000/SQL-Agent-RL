"""Grading semantics, tested on hand-written answers with no DB and no agent."""

from __future__ import annotations

import pytest

from sql_agent_rl.data.answers import (
    AnswerType,
    Entity,
    GroundTruth,
    normalize_entity,
    parse_entity_list,
    parse_numeric,
    score_answer,
)


def _entities(*names: str) -> tuple[Entity, ...]:
    return tuple(Entity.make(name, name) for name in names)


def _ranked(groups: list[list[str]], k: int) -> GroundTruth:
    return GroundTruth(
        answer_type=AnswerType.RANKED_LIST,
        tie_groups=tuple(_entities(*g) for g in groups),
        k=k,
    )


# --------------------------------------------------------------------------
# normalisation and parsing
# --------------------------------------------------------------------------


def test_normalisation_folds_case_accents_and_punctuation() -> None:
    assert normalize_entity("Antônio Carlos Jobim") == normalize_entity(
        "antonio carlos jobim"
    )
    assert normalize_entity("Helena Holý") == normalize_entity("Helena Holy")
    assert normalize_entity("  AC/DC  ") == normalize_entity("ac dc")
    assert normalize_entity("Jane Doe.") == normalize_entity("Jane Doe")


def test_normalisation_does_not_conflate_different_names() -> None:
    assert normalize_entity("Jane Doe") != normalize_entity("John Doe")
    assert normalize_entity("Rock") != normalize_entity("Rock And Roll")


@pytest.mark.parametrize(
    "text,expected",
    [
        ("1234.56", 1234.56),
        ("$1,234.56", 1234.56),
        ("The total was 1234.56 dollars.", 1234.56),
        ("  42 ", 42.0),
        ("1234.56\n", 1234.56),
        ('{"total": 1234.56}', 1234.56),
    ],
)
def test_parse_numeric_accepts_common_shapes(text: str, expected: float) -> None:
    assert parse_numeric(text) == pytest.approx(expected)


def test_parse_numeric_rejects_ambiguous_or_missing() -> None:
    assert parse_numeric("") is None
    assert parse_numeric("no idea") is None
    # Two different numbers: the agent was asked for one figure.
    assert parse_numeric("either 100 or 200") is None


@pytest.mark.parametrize(
    "text",
    [
        "1. Alice Smith\n2. Bob Jones\n3. Carla Diaz",
        "- Alice Smith\n- Bob Jones\n- Carla Diaz",
        "Alice Smith, Bob Jones, Carla Diaz",
        '["Alice Smith", "Bob Jones", "Carla Diaz"]',
        "1. Alice Smith ($49.62)\n2. Bob Jones ($43.62)\n3. Carla Diaz ($39.62)",
        "1. Alice Smith - 49.62\n2. Bob Jones - 43.62\n3. Carla Diaz - 39.62",
    ],
)
def test_parse_entity_list_handles_common_list_shapes(text: str) -> None:
    assert parse_entity_list(text) == ["Alice Smith", "Bob Jones", "Carla Diaz"]


# --------------------------------------------------------------------------
# numeric grading
# --------------------------------------------------------------------------


def test_numeric_tolerance() -> None:
    truth = GroundTruth(answer_type=AnswerType.NUMERIC, value=481.45)
    assert score_answer("481.45", truth) == 1.0
    assert score_answer("481.4500001", truth) == 1.0
    assert score_answer("$481.45", truth) == 1.0
    assert score_answer("481.99", truth) == 0.0
    assert score_answer("nonsense", truth) == 0.0


def test_integer_counts_use_zero_tolerance() -> None:
    truth = GroundTruth(
        answer_type=AnswerType.NUMERIC, value=5.0, abs_tol=0.0, rel_tol=0.0
    )
    assert score_answer("5", truth) == 1.0
    assert score_answer("6", truth) == 0.0


# --------------------------------------------------------------------------
# ranked grading and ties
# --------------------------------------------------------------------------


def test_exact_ranking_scores_one() -> None:
    truth = _ranked([["Alice"], ["Bob"], ["Carla"]], k=3)
    assert score_answer("1. Alice\n2. Bob\n3. Carla", truth) == 1.0


def test_wrong_order_is_penalised_for_ranked_answers() -> None:
    truth = _ranked([["Alice"], ["Bob"], ["Carla"]], k=3)
    # Alice and Carla swapped: only Bob sits in a legitimate position.
    assert score_answer("1. Carla\n2. Bob\n3. Alice", truth) == pytest.approx(1 / 3)


def test_any_linearisation_of_a_tie_group_scores_one() -> None:
    """The tie-handling rule: interchangeable entities really are interchangeable."""
    truth = _ranked([["Alice", "Bob", "Carla"]], k=3)
    for order in [
        "1. Alice\n2. Bob\n3. Carla",
        "1. Carla\n2. Alice\n3. Bob",
        "1. Bob\n2. Carla\n3. Alice",
    ]:
        assert score_answer(order, truth) == 1.0


def test_tie_group_spilling_past_k_accepts_any_member() -> None:
    """Positions 3-5 tie eight ways; naming any one of them fills position 3."""
    truth = _ranked([["Alice"], ["Bob"], ["C", "D", "E", "F", "G", "H"]], k=3)
    assert score_answer("1. Alice\n2. Bob\n3. G", truth) == 1.0
    assert score_answer("1. Alice\n2. Bob\n3. D", truth) == 1.0
    # A tied entity cannot be promoted above the group's rank band.
    assert score_answer("1. G\n2. Alice\n3. Bob", truth) == pytest.approx(0.0)


def test_partial_credit_is_proportional_to_correct_positions() -> None:
    truth = _ranked([["Alice"], ["Bob"], ["Carla"], ["Dan"]], k=4)
    assert score_answer("1. Alice\n2. Bob\n3. Zed\n4. Zoe", truth) == 0.5


def test_repeating_one_correct_name_does_not_farm_credit() -> None:
    truth = _ranked([["Alice"], ["Bob"], ["Carla"]], k=3)
    assert score_answer("1. Alice\n2. Alice\n3. Alice", truth) == pytest.approx(1 / 3)


def test_overlong_answer_is_truncated_to_k() -> None:
    """Listing every customer must not guarantee a hit in every position."""
    truth = _ranked([["Alice"], ["Bob"], ["Carla"]], k=3)
    everyone = "\n".join(f"{i}. {n}" for i, n in enumerate(["Zed", "Zoe", "Zack", "Alice", "Bob", "Carla"], 1))
    assert score_answer(everyone, truth) == 0.0


def test_ids_are_accepted_as_aliases() -> None:
    truth = GroundTruth(
        answer_type=AnswerType.RANKED_LIST,
        tie_groups=(
            (Entity.make(42, "Alice Smith"),),
            (Entity.make(7, "Bob Jones"),),
        ),
        k=2,
    )
    assert score_answer("1. 42\n2. 7", truth) == 1.0
    assert score_answer("1. Alice Smith\n2. Bob Jones", truth) == 1.0


# --------------------------------------------------------------------------
# set grading
# --------------------------------------------------------------------------


def test_exact_set_scores_one_regardless_of_order() -> None:
    truth = GroundTruth(
        answer_type=AnswerType.ENTITY_SET, entities=_entities("Alice", "Bob", "Carla")
    )
    assert score_answer("Carla\nAlice\nBob", truth) == 1.0


def test_set_uses_jaccard_so_dumping_everything_scores_near_zero() -> None:
    """Recall alone would reward listing the whole catalogue; Jaccard does not."""
    truth = GroundTruth(
        answer_type=AnswerType.ENTITY_SET, entities=_entities("Alice", "Bob")
    )
    dump = "\n".join(["Alice", "Bob"] + [f"Filler {i}" for i in range(48)])
    assert score_answer(dump, truth) == pytest.approx(2 / 50)
    assert score_answer("Alice\nBob", truth) == 1.0


def test_set_partial_credit() -> None:
    truth = GroundTruth(
        answer_type=AnswerType.ENTITY_SET, entities=_entities("Alice", "Bob", "Carla")
    )
    assert score_answer("Alice\nBob", truth) == pytest.approx(2 / 3)
    assert score_answer("Alice\nZed", truth) == pytest.approx(1 / 4)


def test_duplicate_submissions_do_not_inflate_set_score() -> None:
    truth = GroundTruth(
        answer_type=AnswerType.ENTITY_SET, entities=_entities("Alice", "Bob")
    )
    assert score_answer("Alice\nAlice\nAlice", truth) == pytest.approx(1 / 2)


# --------------------------------------------------------------------------
# categorical grading
# --------------------------------------------------------------------------


def _categorical(winners: list[str], others: list[str]) -> GroundTruth:
    return GroundTruth(
        answer_type=AnswerType.CATEGORICAL,
        tie_groups=(_entities(*winners),),
        k=1,
        distractors=_entities(*others),
    )


def test_categorical_exact_and_prose_answers() -> None:
    truth = _categorical(["Rock"], ["Latin", "Jazz", "Metal"])
    assert score_answer("Rock", truth) == 1.0
    assert score_answer("The top genre was Rock.", truth) == 1.0
    assert score_answer("Latin", truth) == 0.0


def test_categorical_rejects_hedging_across_several_candidates() -> None:
    truth = _categorical(["Rock"], ["Latin", "Jazz", "Metal"])
    assert score_answer("Rock, Latin", truth) == 0.0
    assert score_answer("It is either Rock or Latin or Jazz.", truth) == 0.0


def test_categorical_accepts_any_genuinely_tied_winner() -> None:
    truth = _categorical(["Rock", "Latin"], ["Jazz"])
    assert score_answer("Rock", truth) == 1.0
    assert score_answer("Latin", truth) == 1.0
    assert score_answer("Jazz", truth) == 0.0


def test_empty_answers_score_zero_everywhere() -> None:
    for truth in [
        _ranked([["Alice"]], k=1),
        GroundTruth(answer_type=AnswerType.ENTITY_SET, entities=_entities("Alice")),
        GroundTruth(answer_type=AnswerType.NUMERIC, value=1.0),
        _categorical(["Rock"], ["Jazz"]),
    ]:
        assert score_answer("", truth) == 0.0
        assert score_answer("   ", truth) == 0.0
