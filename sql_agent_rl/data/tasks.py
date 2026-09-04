"""Parameterised task templates and their pandas ground truth.

Because Chinook is a *fixed* real database rather than synthetic data with a
seed, task diversity has to come from this layer instead of the data layer.
Each template samples its parameters (which year, which country, which genre,
which K) from values discovered in the database at runtime, so an agent cannot
memorise a single question-answer pair -- and no year, country or genre is ever
hardcoded, since upstream Chinook shifts its invoice dates forward over time.

Every ``compute_ground_truth`` is a pandas aggregation over
:class:`~sql_agent_rl.data.chinook.ChinookFrames`. None of them runs SQL. That
is deliberate: grading against a reference query's output would only test SQL
equivalence to one particular solution, not correctness.

Documented conventions
----------------------
Revenue
    ``SUM(InvoiceLine.UnitPrice * InvoiceLine.Quantity)``, rounded half-up to
    cents at the end of the aggregation. See :mod:`sql_agent_rl.data.chinook`.
Customer country
    ``Customer.Country``. ``Invoice.BillingCountry`` agrees on every row of
    this build (verified in the test suite), so either join works.
Zero-revenue periods
    A customer with no invoices in a period has revenue 0.00 for that period,
    not NULL. This matters for the year-over-year change template: 50 of the
    59 customers are missing from at least one year.
Ties
    Ranked answers are graded against ordered tie groups, so any linearisation
    consistent with the true ordering scores 1.0. See
    :mod:`sql_agent_rl.data.answers`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Sequence

import pandas as pd

from sql_agent_rl.data.answers import AnswerType, Entity, GroundTruth
from sql_agent_rl.data.chinook import ChinookFrames, round_currency


class Difficulty(str, Enum):
    """Tiers, defined by how much joining and reshaping the question needs."""

    #: One table, one aggregate.
    SINGLE_TABLE = "single_table"
    #: A 3-5 table join through the sales fact path.
    MULTI_JOIN = "multi_join"
    #: A join *plus* a comparison across two time windows, or an anti-join.
    TIME_COMPARISON = "time_comparison"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


@dataclass(frozen=True)
class Task:
    """One concrete, fully-parameterised episode."""

    template_id: str
    difficulty: Difficulty
    question: str
    answer_format: str
    ground_truth: GroundTruth
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def prompt(self) -> str:
        return f"{self.question}\n\nAnswer format: {self.answer_format}"


class TaskTemplate:
    """Base class: sample parameters, render a question, compute truth."""

    template_id: str = ""
    difficulty: Difficulty = Difficulty.MULTI_JOIN

    def sample_params(self, frames: ChinookFrames, rng: random.Random) -> dict[str, Any]:
        raise NotImplementedError

    def build(self, frames: ChinookFrames, params: dict[str, Any]) -> Task | None:
        """Return a Task, or ``None`` if this parameterisation is degenerate.

        Degenerate means the question has no meaningful answer -- an empty
        result, or fewer entities than the requested K. The sampler retries.
        """
        raise NotImplementedError


# --------------------------------------------------------------------------
# helpers shared by templates
# --------------------------------------------------------------------------


def _tie_groups_from_series(
    ranked: pd.Series,
    k: int,
    make_entity: Callable[[Any], Entity],
) -> tuple[tuple[Entity, ...], ...]:
    """Turn a value-sorted Series into ordered tie groups covering the top K.

    The final group is kept *whole* even if it spills past K: those entities
    are genuinely interchangeable at the cut-off, and any of them is an
    acceptable occupant of the remaining positions.
    """
    groups: list[tuple[Entity, ...]] = []
    covered = 0
    for _value, chunk in _group_by_value(ranked):
        groups.append(tuple(make_entity(idx) for idx in chunk))
        covered += len(chunk)
        if covered >= k:
            break
    return tuple(groups)


def _group_by_value(ranked: pd.Series) -> list[tuple[float, list[Any]]]:
    """Consecutive runs of equal value in an already-sorted Series."""
    out: list[tuple[float, list[Any]]] = []
    for index, value in ranked.items():
        if out and out[-1][0] == value:
            out[-1][1].append(index)
        else:
            out.append((value, [index]))
    return out


def _customer_entity(frames: ChinookFrames) -> Callable[[Any], Entity]:
    names = frames.customer.set_index("CustomerId")
    def make(customer_id: Any) -> Entity:
        row = names.loc[customer_id]
        display = f"{row['FirstName']} {row['LastName']}"
        return Entity.make(customer_id, display, row["LastName"])
    return make


def _plain_entity(prefix: str = "") -> Callable[[Any], Entity]:
    def make(value: Any) -> Entity:
        return Entity.make(f"{prefix}{value}", str(value))
    return make


def _year_revenue(frames: ChinookFrames, year: int) -> pd.DataFrame:
    return frames.sales[frames.sales["year"] == year]


# --------------------------------------------------------------------------
# tier 1 -- single-table aggregation
# --------------------------------------------------------------------------


class CustomerCountInCountry(TaskTemplate):
    """Customer only. Tests that the agent can find and filter one table."""

    template_id = "customer_count_in_country"
    difficulty = Difficulty.SINGLE_TABLE

    def sample_params(self, frames, rng):
        return {"country": rng.choice(frames.countries)}

    def build(self, frames, params):
        country = params["country"]
        count = int((frames.customer["Country"] == country).sum())
        if count == 0:
            return None
        return Task(
            template_id=self.template_id,
            difficulty=self.difficulty,
            question=f"How many customers are based in {country}?",
            answer_format="a single integer, e.g. `7`",
            ground_truth=GroundTruth(
                answer_type=AnswerType.NUMERIC,
                value=float(count),
                abs_tol=0.0,
                rel_tol=0.0,
                notes=f"{count} customers in {country}",
            ),
            params=params,
        )


class TotalRevenueInYear(TaskTemplate):
    """Invoice + InvoiceLine. The simplest revenue question."""

    template_id = "total_revenue_in_year"
    difficulty = Difficulty.SINGLE_TABLE

    def sample_params(self, frames, rng):
        return {"year": rng.choice(frames.years)}

    def build(self, frames, params):
        year = params["year"]
        subset = _year_revenue(frames, year)
        if subset.empty:
            return None
        total = round_currency(subset["revenue"].sum())
        return Task(
            template_id=self.template_id,
            difficulty=self.difficulty,
            question=(
                f"What was the total revenue from all track sales in {year}? "
                "Revenue means the sum of UnitPrice multiplied by Quantity over "
                "invoice line items."
            ),
            answer_format="a single number rounded to 2 decimals, e.g. `1234.56`",
            ground_truth=GroundTruth(
                answer_type=AnswerType.NUMERIC,
                value=total,
                notes=f"total revenue {year} = {total}",
            ),
            params=params,
        )


class TracksSoldInGenreYear(TaskTemplate):
    """Genre + Track + InvoiceLine + Invoice, but only a COUNT."""

    template_id = "tracks_sold_in_genre_year"
    difficulty = Difficulty.SINGLE_TABLE

    def sample_params(self, frames, rng):
        return {"genre": rng.choice(frames.genres), "year": rng.choice(frames.years)}

    def build(self, frames, params):
        genre, year = params["genre"], params["year"]
        subset = _year_revenue(frames, year)
        count = int(subset.loc[subset["GenreName"] == genre, "Quantity"].sum())
        if count == 0:
            return None
        return Task(
            template_id=self.template_id,
            difficulty=self.difficulty,
            question=(
                f"How many individual tracks in the {genre} genre were sold in {year}? "
                "Count the total quantity across invoice line items."
            ),
            answer_format="a single integer, e.g. `142`",
            ground_truth=GroundTruth(
                answer_type=AnswerType.NUMERIC,
                value=float(count),
                abs_tol=0.0,
                rel_tol=0.0,
                notes=f"{count} {genre} tracks sold in {year}",
            ),
            params=params,
        )


# --------------------------------------------------------------------------
# tier 2 -- multi-table joins
# --------------------------------------------------------------------------


class TopCustomersByRevenue(TaskTemplate):
    """Customer -> Invoice -> InvoiceLine, ranked."""

    template_id = "top_customers_by_revenue"
    difficulty = Difficulty.MULTI_JOIN

    def sample_params(self, frames, rng):
        return {"year": rng.choice(frames.years), "k": rng.choice([3, 5])}

    def build(self, frames, params):
        year, k = params["year"], params["k"]
        subset = _year_revenue(frames, year)
        ranked = (
            subset.groupby("CustomerId")["revenue"]
            .sum()
            .map(round_currency)
            .sort_values(ascending=False, kind="mergesort")
        )
        if len(ranked) < k:
            return None
        groups = _tie_groups_from_series(ranked, k, _customer_entity(frames))
        return Task(
            template_id=self.template_id,
            difficulty=self.difficulty,
            question=(
                f"Which {k} customers generated the most revenue in {year}? "
                "List them from highest to lowest revenue. Revenue means the sum of "
                "UnitPrice multiplied by Quantity over their invoice line items."
            ),
            answer_format=(
                f"a numbered list of exactly {k} customer full names, highest first, "
                "one per line, e.g. `1. Jane Doe`"
            ),
            ground_truth=GroundTruth(
                answer_type=AnswerType.RANKED_LIST,
                tie_groups=groups,
                k=k,
                notes=f"top {k} customers by revenue in {year}",
            ),
            params=params,
        )


class TopArtistsByRevenue(TaskTemplate):
    """The full fact path: Invoice -> InvoiceLine -> Track -> Album -> Artist."""

    template_id = "top_artists_by_revenue"
    difficulty = Difficulty.MULTI_JOIN

    def sample_params(self, frames, rng):
        return {"year": rng.choice(frames.years), "k": rng.choice([3, 5])}

    def build(self, frames, params):
        year, k = params["year"], params["k"]
        subset = _year_revenue(frames, year).dropna(subset=["ArtistName"])
        ranked = (
            subset.groupby("ArtistName")["revenue"]
            .sum()
            .map(round_currency)
            .sort_values(ascending=False, kind="mergesort")
        )
        if len(ranked) < k:
            return None
        groups = _tie_groups_from_series(ranked, k, _plain_entity())
        return Task(
            template_id=self.template_id,
            difficulty=self.difficulty,
            question=(
                f"Which {k} artists generated the most revenue in {year}? "
                "Attribute a track's revenue to the artist of its album. "
                "List them from highest to lowest revenue."
            ),
            answer_format=(
                f"a numbered list of exactly {k} artist names, highest first, "
                "one per line, e.g. `1. Some Artist`"
            ),
            ground_truth=GroundTruth(
                answer_type=AnswerType.RANKED_LIST,
                tie_groups=groups,
                k=k,
                notes=f"top {k} artists by revenue in {year}",
            ),
            params=params,
        )


class TopGenreInCountry(TaskTemplate):
    """Customer -> Invoice -> InvoiceLine -> Track -> Genre, single winner."""

    template_id = "top_genre_in_country"
    difficulty = Difficulty.MULTI_JOIN

    def sample_params(self, frames, rng):
        counts = frames.customer["Country"].value_counts()
        eligible = counts[counts >= 2].index.tolist() or frames.countries
        return {"country": rng.choice(sorted(eligible))}

    def build(self, frames, params):
        country = params["country"]
        subset = frames.sales[frames.sales["Country"] == country].dropna(
            subset=["GenreName"]
        )
        if subset.empty:
            return None
        ranked = (
            subset.groupby("GenreName")["revenue"]
            .sum()
            .map(round_currency)
            .sort_values(ascending=False, kind="mergesort")
        )
        winners = ranked[ranked == ranked.iloc[0]].index.tolist()
        make = _plain_entity()
        others = tuple(make(g) for g in frames.genres if g not in winners)
        return Task(
            template_id=self.template_id,
            difficulty=self.difficulty,
            question=(
                f"Which music genre generated the most total revenue from customers "
                f"based in {country}, across all years?"
            ),
            answer_format="the genre name only, e.g. `Rock`",
            ground_truth=GroundTruth(
                answer_type=AnswerType.CATEGORICAL,
                tie_groups=(tuple(make(g) for g in winners),),
                k=1,
                distractors=others,
                notes=f"top genre in {country}: {', '.join(winners)}",
            ),
            params=params,
        )


class TopCountriesByRevenue(TaskTemplate):
    """Customer -> Invoice -> InvoiceLine, grouped by country."""

    template_id = "top_countries_by_revenue"
    difficulty = Difficulty.MULTI_JOIN

    def sample_params(self, frames, rng):
        return {"year": rng.choice(frames.years), "k": rng.choice([3, 5])}

    def build(self, frames, params):
        year, k = params["year"], params["k"]
        subset = _year_revenue(frames, year).dropna(subset=["Country"])
        ranked = (
            subset.groupby("Country")["revenue"]
            .sum()
            .map(round_currency)
            .sort_values(ascending=False, kind="mergesort")
        )
        if len(ranked) < k:
            return None
        groups = _tie_groups_from_series(ranked, k, _plain_entity())
        return Task(
            template_id=self.template_id,
            difficulty=self.difficulty,
            question=(
                f"Which {k} countries generated the most revenue in {year}? "
                "Use the customer's country. List them from highest to lowest revenue."
            ),
            answer_format=(
                f"a numbered list of exactly {k} country names, highest first, "
                "one per line, e.g. `1. Canada`"
            ),
            ground_truth=GroundTruth(
                answer_type=AnswerType.RANKED_LIST,
                tie_groups=groups,
                k=k,
                notes=f"top {k} countries by revenue in {year}",
            ),
            params=params,
        )


# --------------------------------------------------------------------------
# tier 3 -- time-window comparisons and anti-joins
# --------------------------------------------------------------------------


class TopCustomerRevenueIncrease(TaskTemplate):
    """The headline task: compare two years per customer, rank the deltas.

    Customers absent from either year count as 0.00 for that year -- the
    edge case that makes a naive inner-join solution wrong.
    """

    template_id = "top_customer_revenue_increase"
    difficulty = Difficulty.TIME_COMPARISON

    def sample_params(self, frames, rng):
        years = frames.years
        if len(years) < 2:
            return {}
        year_a = rng.choice(years[:-1])
        year_b = rng.choice([y for y in years if y > year_a])
        return {"year_a": year_a, "year_b": year_b, "k": rng.choice([3, 5])}

    def build(self, frames, params):
        if not params:
            return None
        year_a, year_b, k = params["year_a"], params["year_b"], params["k"]
        per_year = (
            frames.sales[frames.sales["year"].isin([year_a, year_b])]
            .groupby(["CustomerId", "year"])["revenue"]
            .sum()
            .unstack("year")
        )
        # Reindex over *all* customers, then fill: a customer with no invoices
        # in a year earned 0.00 that year, which is not the same as NULL.
        per_year = per_year.reindex(frames.customer["CustomerId"]).fillna(0.0)
        for year in (year_a, year_b):
            if year not in per_year.columns:
                per_year[year] = 0.0
        delta = (
            (per_year[year_b].map(round_currency) - per_year[year_a].map(round_currency))
            .map(round_currency)
            .sort_values(ascending=False, kind="mergesort")
        )
        if len(delta) < k or delta.iloc[0] <= 0:
            return None
        groups = _tie_groups_from_series(delta, k, _customer_entity(frames))
        return Task(
            template_id=self.template_id,
            difficulty=self.difficulty,
            question=(
                f"Which {k} customers had the largest increase in revenue from "
                f"{year_a} to {year_b}? Increase means {year_b} revenue minus "
                f"{year_a} revenue; a customer who bought nothing in a year counts "
                f"as 0.00 for that year. List them from largest increase to smallest."
            ),
            answer_format=(
                f"a numbered list of exactly {k} customer full names, largest "
                "increase first, one per line, e.g. `1. Jane Doe`"
            ),
            ground_truth=GroundTruth(
                answer_type=AnswerType.RANKED_LIST,
                tie_groups=groups,
                k=k,
                notes=f"largest revenue increases {year_a}->{year_b}",
            ),
            params=params,
        )


class ArtistsWithNoSalesInYear(TaskTemplate):
    """An anti-join scoped to one genre, so the answer set stays reviewable."""

    template_id = "artists_with_no_sales_in_year"
    difficulty = Difficulty.TIME_COMPARISON

    #: Answer sets outside this band are rejected: an empty set is a trick
    #: question, and a 60-name set is a transcription exercise, not analysis.
    MIN_ANSWERS = 1
    MAX_ANSWERS = 15

    def sample_params(self, frames, rng):
        return {"genre": rng.choice(frames.genres), "year": rng.choice(frames.years)}

    def build(self, frames, params):
        genre, year = params["genre"], params["year"]
        in_genre = frames.sales[frames.sales["GenreName"] == genre]
        catalogue = self._catalogue_artists(frames, genre)
        if len(catalogue) < 2:
            return None
        sold = set(in_genre.loc[in_genre["year"] == year, "ArtistName"].dropna())
        silent = sorted(catalogue - sold)
        if not (self.MIN_ANSWERS <= len(silent) <= self.MAX_ANSWERS):
            return None
        make = _plain_entity()
        return Task(
            template_id=self.template_id,
            difficulty=self.difficulty,
            question=(
                f"Consider every artist who has at least one {genre} track in the "
                f"catalogue. Which of them sold no {genre} tracks at all in {year}?"
            ),
            answer_format=(
                "a list of artist names, one per line; order does not matter"
            ),
            ground_truth=GroundTruth(
                answer_type=AnswerType.ENTITY_SET,
                entities=tuple(make(a) for a in silent),
                notes=f"{len(silent)} {genre} artists with no sales in {year}",
            ),
            params=params,
        )

    @staticmethod
    def _catalogue_artists(frames: ChinookFrames, genre: str) -> set[str]:
        """Artists with a track in this genre, whether or not it ever sold."""
        genre_ids = frames.genre.loc[frames.genre["Name"] == genre, "GenreId"]
        tracks = frames.track[frames.track["GenreId"].isin(genre_ids)]
        albums = frames.album[frames.album["AlbumId"].isin(tracks["AlbumId"])]
        artists = frames.artist[frames.artist["ArtistId"].isin(albums["ArtistId"])]
        return set(artists["Name"].dropna())


# --------------------------------------------------------------------------
# registry and sampling
# --------------------------------------------------------------------------

TEMPLATES: tuple[TaskTemplate, ...] = (
    CustomerCountInCountry(),
    TotalRevenueInYear(),
    TracksSoldInGenreYear(),
    TopCustomersByRevenue(),
    TopArtistsByRevenue(),
    TopGenreInCountry(),
    TopCountriesByRevenue(),
    TopCustomerRevenueIncrease(),
    ArtistsWithNoSalesInYear(),
)

TEMPLATES_BY_ID = {t.template_id: t for t in TEMPLATES}


def sample_task(
    frames: ChinookFrames,
    rng: random.Random,
    template_ids: Sequence[str] | None = None,
    difficulties: Sequence[Difficulty | str] | None = None,
    max_attempts: int = 50,
) -> Task:
    """Draw one viable task, retrying degenerate parameterisations."""
    pool = _filter_templates(template_ids, difficulties)
    for _ in range(max_attempts):
        template = rng.choice(pool)
        task = template.build(frames, template.sample_params(frames, rng))
        if task is not None:
            return task
    raise RuntimeError(
        f"Could not sample a viable task in {max_attempts} attempts "
        f"(templates={[t.template_id for t in pool]})."
    )


def generate_tasks(
    frames: ChinookFrames,
    n: int,
    seed: int = 0,
    template_ids: Sequence[str] | None = None,
    difficulties: Sequence[Difficulty | str] | None = None,
) -> list[Task]:
    """Draw ``n`` tasks. Deterministic given ``seed`` and the fixed database."""
    rng = random.Random(seed)
    return [
        sample_task(frames, rng, template_ids, difficulties) for _ in range(n)
    ]


def _filter_templates(
    template_ids: Sequence[str] | None,
    difficulties: Sequence[Difficulty | str] | None,
) -> list[TaskTemplate]:
    pool = list(TEMPLATES)
    if template_ids:
        unknown = set(template_ids) - set(TEMPLATES_BY_ID)
        if unknown:
            raise KeyError(f"Unknown template id(s): {sorted(unknown)}")
        pool = [TEMPLATES_BY_ID[tid] for tid in template_ids]
    if difficulties:
        wanted = {str(d) for d in difficulties}
        pool = [t for t in pool if str(t.difficulty) in wanted]
    if not pool:
        raise ValueError("No task templates match the requested filters.")
    return pool
