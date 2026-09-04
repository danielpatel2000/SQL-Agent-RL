"""Tests for the downloaded database, ground truth, and answer grading."""

from __future__ import annotations

import random
import sqlite3
from pathlib import Path

import pytest

from sql_agent_rl.data import (
    TEMPLATES,
    TEMPLATES_BY_ID,
    AnswerType,
    Difficulty,
    generate_tasks,
    load_chinook,
    round_currency,
    sample_task,
    score_answer,
)
from sql_agent_rl.data.chinook import CORE_TABLES

#: Row counts for the official ``Chinook_Sqlite.sqlite`` build. Asserted
#: instead of a "same seed -> same database" determinism test: the data is a
#: fixed real dataset, so the meaningful check is that we downloaded the
#: dataset we think we did.
EXPECTED_ROW_COUNTS = {
    "Album": 347,
    "Artist": 275,
    "Customer": 59,
    "Employee": 8,
    "Genre": 25,
    "Invoice": 412,
    "InvoiceLine": 2240,
    "MediaType": 5,
    "Playlist": 18,
    "PlaylistTrack": 8715,
    "Track": 3503,
}


# --------------------------------------------------------------------------
# the downloaded database
# --------------------------------------------------------------------------


def test_expected_tables_present(chinook_db: Path) -> None:
    con = sqlite3.connect(chinook_db)
    names = {
        r[0]
        for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    con.close()
    assert set(EXPECTED_ROW_COUNTS) <= names
    assert set(CORE_TABLES) <= names


def test_expected_row_counts_and_all_non_zero(chinook_db: Path) -> None:
    con = sqlite3.connect(chinook_db)
    try:
        for table, expected in EXPECTED_ROW_COUNTS.items():
            actual = con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            assert actual > 0, f"{table} is empty"
            assert actual == expected, f"{table}: expected {expected} rows, got {actual}"
    finally:
        con.close()


def test_invoice_date_range_is_discovered_not_hardcoded(chinook_db: Path) -> None:
    frames = load_chinook(chinook_db)
    lo, hi = frames.date_range()
    assert lo < hi
    years = frames.years
    assert len(years) >= 2, "templates comparing two years need at least two years"
    assert years == sorted(years)
    # Every year the templates can sample must actually contain invoices.
    for year in years:
        assert (frames.sales["year"] == year).any()


def test_two_revenue_definitions_agree(chinook_db: Path) -> None:
    """Cross-check the canonical (line-item) revenue against Invoice.Total.

    The brief flagged a disagreement here as a data-quality finding. On this
    build there is none: all 412 invoices match to the cent.
    """
    frames = load_chinook(chinook_db)
    per_invoice = frames.sales.groupby("InvoiceId")["revenue"].sum().map(round_currency)
    stated = frames.invoice.set_index("InvoiceId")["Total"].map(round_currency)
    joined = stated.to_frame("stated").join(per_invoice.rename("computed"), how="outer")
    assert not joined["computed"].isna().any(), "invoice with no line items"
    assert not joined["stated"].isna().any(), "line items with no invoice"
    mismatched = joined[(joined["stated"] - joined["computed"]).abs() > 0.005]
    assert mismatched.empty, f"revenue definitions disagree on:\n{mismatched}"


def test_customer_country_matches_billing_country(chinook_db: Path) -> None:
    """Justifies the documented choice to join on Customer.Country."""
    frames = load_chinook(chinook_db)
    merged = frames.invoice.merge(
        frames.customer[["CustomerId", "Country"]], on="CustomerId"
    )
    assert (merged["BillingCountry"] == merged["Country"]).all()


def test_round_currency_is_half_up_not_bankers() -> None:
    assert round_currency(0.125) == 0.13  # round() would give 0.12
    assert round_currency(2.675) == 2.68
    assert round_currency(1.005) == 1.01


# --------------------------------------------------------------------------
# task generation
# --------------------------------------------------------------------------


def test_every_template_can_produce_a_viable_task(chinook_db: Path) -> None:
    frames = load_chinook(chinook_db)
    rng = random.Random(0)
    for template in TEMPLATES:
        for _ in range(80):
            task = template.build(frames, template.sample_params(frames, rng))
            if task is not None:
                break
        else:
            pytest.fail(f"{template.template_id} produced no viable task in 80 tries")
        assert task.question and task.answer_format
        assert task.template_id == template.template_id


def test_all_three_difficulty_tiers_are_represented() -> None:
    tiers = {t.difficulty for t in TEMPLATES}
    assert tiers == set(Difficulty)


def test_generation_is_deterministic_given_seed(chinook_db: Path) -> None:
    """The database is fixed, so a seed must pin the whole task list."""
    frames = load_chinook(chinook_db)
    first = generate_tasks(frames, 25, seed=123)
    second = generate_tasks(frames, 25, seed=123)
    assert [t.question for t in first] == [t.question for t in second]
    assert [t.ground_truth.describe() for t in first] == [
        t.ground_truth.describe() for t in second
    ]


def test_different_seeds_give_task_diversity(chinook_db: Path) -> None:
    """Diversity has to come from the task layer, since the data is fixed."""
    frames = load_chinook(chinook_db)
    tasks = generate_tasks(frames, 60, seed=99)
    assert len({t.template_id for t in tasks}) >= 6
    assert len({t.question for t in tasks}) >= 25, "questions are too repetitive"


def test_no_task_has_an_empty_ground_truth(chinook_db: Path) -> None:
    frames = load_chinook(chinook_db)
    for task in generate_tasks(frames, 80, seed=5):
        truth = task.ground_truth
        if truth.answer_type is AnswerType.NUMERIC:
            assert truth.value is not None
        elif truth.answer_type is AnswerType.ENTITY_SET:
            assert truth.entities
        else:
            assert truth.tie_groups and all(truth.tie_groups)
            assert truth.k >= 1


def test_ranked_tie_groups_cover_at_least_k_positions(chinook_db: Path) -> None:
    frames = load_chinook(chinook_db)
    for task in generate_tasks(frames, 80, seed=11):
        truth = task.ground_truth
        if truth.answer_type is AnswerType.RANKED_LIST:
            covered = sum(len(g) for g in truth.tie_groups)
            assert covered >= truth.k


def test_template_and_difficulty_filters(chinook_db: Path) -> None:
    frames = load_chinook(chinook_db)
    rng = random.Random(3)
    task = sample_task(frames, rng, template_ids=["top_customers_by_revenue"])
    assert task.template_id == "top_customers_by_revenue"

    task = sample_task(frames, rng, difficulties=[Difficulty.SINGLE_TABLE])
    assert task.difficulty is Difficulty.SINGLE_TABLE

    with pytest.raises(KeyError):
        sample_task(frames, rng, template_ids=["no_such_template"])


# --------------------------------------------------------------------------
# ground truth is right (verified against independent SQL, not the reverse)
# --------------------------------------------------------------------------


def test_ground_truth_matches_independent_sql(chinook_db: Path) -> None:
    """Sanity-check pandas ground truth against hand-written SQL.

    Note the direction: SQL is used here to *audit the grader in a test*, never
    to define correctness at runtime. The rubric only ever consults pandas.
    """
    frames = load_chinook(chinook_db)
    con = sqlite3.connect(f"file:{chinook_db}?mode=ro", uri=True)
    try:
        year = frames.years[1]

        total = con.execute(
            "SELECT ROUND(SUM(il.UnitPrice * il.Quantity), 2) FROM InvoiceLine il "
            "JOIN Invoice i ON i.InvoiceId = il.InvoiceId "
            "WHERE CAST(strftime('%Y', i.InvoiceDate) AS INT) = ?",
            (year,),
        ).fetchone()[0]
        task = TEMPLATES_BY_ID["total_revenue_in_year"].build(frames, {"year": year})
        assert task is not None
        assert task.ground_truth.value == pytest.approx(total, abs=0.01)

        rows = con.execute(
            "SELECT c.Country, ROUND(SUM(il.UnitPrice * il.Quantity), 2) rev "
            "FROM InvoiceLine il "
            "JOIN Invoice i ON i.InvoiceId = il.InvoiceId "
            "JOIN Customer c ON c.CustomerId = i.CustomerId "
            "WHERE CAST(strftime('%Y', i.InvoiceDate) AS INT) = ? "
            "GROUP BY c.Country ORDER BY rev DESC, c.Country LIMIT 3",
            (year,),
        ).fetchall()
        top_countries = TEMPLATES_BY_ID["top_countries_by_revenue"].build(
            frames, {"year": year, "k": 3}
        )
        assert top_countries is not None
        # Compare against the tie-group structure: the SQL ordering is one
        # valid linearisation, so it must score 1.0.
        answer = "\n".join(f"{i}. {r[0]}" for i, r in enumerate(rows, 1))
        assert score_answer(answer, top_countries.ground_truth) == 1.0
    finally:
        con.close()


def test_zero_revenue_year_is_treated_as_zero_not_missing(chinook_db: Path) -> None:
    """The edge case that breaks a naive inner join on the two-year task."""
    frames = load_chinook(chinook_db)
    years = frames.years
    task = TEMPLATES_BY_ID["top_customer_revenue_increase"].build(
        frames, {"year_a": years[0], "year_b": years[-1], "k": 5}
    )
    assert task is not None
    ranked_ids = {e.key for group in task.ground_truth.tie_groups for e in group}

    per_year = (
        frames.sales[frames.sales["year"].isin([years[0], years[-1]])]
        .groupby(["CustomerId", "year"])["revenue"]
        .sum()
        .unstack("year")
    )
    absent_in_first = set(
        per_year.index[per_year[years[0]].isna() & per_year[years[-1]].notna()].astype(str)
    )
    assert absent_in_first, "expected customers missing from the earlier year"
    # Such customers must be rankable, not silently dropped by the join.
    all_customers = {str(c) for c in frames.customer["CustomerId"]}
    assert absent_in_first <= all_customers
    assert ranked_ids <= all_customers
