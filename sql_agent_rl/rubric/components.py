"""Reward components: pure functions of ``(Trajectory, GroundTruth)``.

Each returns an unweighted magnitude in ``[0, 1]``; the weights in
:mod:`sql_agent_rl.rubric.reward` turn those into signed contributions. Keeping
magnitude and sign apart means a component can be read, tested and re-weighted
without touching its logic, and it keeps the "penalties are small" claim
checkable at a glance.

Nothing here touches a database, a model, or an RL framework, so every
component -- including the anti-hacking ones -- is unit-testable against
hand-written trajectories.
"""

from __future__ import annotations

from sql_agent_rl.data.answers import (
    AnswerType,
    GroundTruth,
    normalize_entity,
    parse_entity_list,
    parse_numeric,
    score_answer,
)
from sql_agent_rl.env.trajectory import TerminationReason, Trajectory

#: Errors beyond this many stop adding penalty. The cap is load-bearing: it is
#: what keeps "tried hard and failed" strictly better than "never tried".
MAX_PENALISED_ERRORS = 5

#: Tool calls that incur no efficiency pressure.
FREE_TOOL_CALLS = 8

#: Calls beyond ``FREE_TOOL_CALLS`` that still add penalty.
MAX_PENALISED_EXCESS_CALLS = 5


# --------------------------------------------------------------------------
# 1. correctness
# --------------------------------------------------------------------------


def correctness(trajectory: Trajectory, truth: GroundTruth) -> float:
    """How close the submitted answer is to the pandas-computed ground truth.

    Compared against ground truth recomputed from the *full* dataset -- never
    against the rows the agent happened to retrieve, and never against a
    reference query's output. An answer that matches a truncated result set by
    omission is simply wrong here.
    """
    if trajectory.submitted_answer is None:
        return 0.0
    return score_answer(trajectory.submitted_answer, truth)


# --------------------------------------------------------------------------
# 2. grounding (a gate, not an addend)
# --------------------------------------------------------------------------


def grounding(trajectory: Trajectory, truth: GroundTruth) -> float:
    """1.0 when the answer is traceable to real query results, else 0.0.

    Two conditions, both required:

    a. **A real query ran.** At least one successful ``execute_sql`` whose
       execution actually read a data table. This is decided by SQLite's own
       authorizer callback, not by pattern-matching table names in the SQL
       text, so ``SELECT 'Acme Corp' AS name`` and ``SELECT * FROM
       sqlite_master`` both fail it -- they execute fine but read no data.

    b. **The answer appears in what came back.** At least one submitted value
       occurs in a result row the agent actually saw.

    Condition (b) is intentionally lenient (*at least one* value, matched
    against the concatenated text of a row so a name split across FirstName and
    LastName still counts). Results are truncated to 50 rows, so demanding that
    *every* submitted value be present would fail honest agents whose answer
    sits below the cut. Correctness is the main defence against guessing --
    landing a ranked top-5 by chance out of 59 customers is a 1-in-5-million
    event -- and grounding is the backstop that stops an ungrounded answer from
    ever collecting that reward.
    """
    if trajectory.submitted_answer is None:
        return 0.0
    if not trajectory.grounded_queries:
        return 0.0
    return 1.0 if _answer_is_traceable(trajectory, truth) else 0.0


def _answer_is_traceable(trajectory: Trajectory, truth: GroundTruth) -> bool:
    answer = trajectory.submitted_answer or ""
    rows = trajectory.result_rows()
    if not rows:
        return False

    if truth.answer_type is AnswerType.NUMERIC:
        submitted = parse_numeric(answer)
        if submitted is None:
            return False
        tolerance = max(truth.abs_tol, 0.01)
        for row in rows:
            for cell in row:
                value = _as_float(cell)
                if value is not None and abs(value - submitted) <= tolerance:
                    return True
        return False

    # Match against the whole row's text, so a customer named across two
    # columns ("Leonie", "Köhler") still matches the submitted "Leonie Köhler".
    row_texts = [
        normalize_entity(" ".join("" if c is None else str(c) for c in row))
        for row in rows
    ]
    for item in parse_entity_list(answer):
        needle = normalize_entity(item)
        if needle and any(needle in text for text in row_texts):
            return True
    return False


def _as_float(cell: object) -> float | None:
    if isinstance(cell, bool):
        return None
    if isinstance(cell, (int, float)):
        return float(cell)
    try:
        return float(str(cell).replace(",", "").replace("$", ""))
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------
# 3. process shaping (small, capped)
# --------------------------------------------------------------------------


def schema_first_bonus(trajectory: Trajectory, truth: GroundTruth) -> float:
    """1.0 if the agent inspected the schema before its first query.

    Awarded **once**, and only for inspection that precedes the first query.
    Both halves matter: a per-call bonus would be farmable by spamming
    ``inspect_schema``, and a bonus for inspecting at any point would reward
    looking at the schema after already guessing at column names.
    """
    del truth
    return 1.0 if trajectory.inspected_schema_before_first_query else 0.0


def wasted_call_penalty(trajectory: Trajectory, truth: GroundTruth) -> float:
    """Fraction of the wasted-call cap incurred, in ``[0, 1]``.

    A call is wasted when it returns no new information:

    * a **failed query** -- the error is the signal to fix and retry;
    * a **degenerate query** that succeeded without reading a data table,
      which closes the "burn calls on ``SELECT 1`` for free" loophole;
    * a **redundant schema inspection** -- the schema is immutable within an
      episode, so repeats return identical text. Without this, farming the
      schema bonus is merely capped rather than loss-making: six inspections
      plus a query plus a submit fit inside the free call budget exactly, and
      would otherwise score identically to inspecting once.

    The total is capped rather than unbounded so that self-correction stays
    affordable: an agent that hits an error, reads it, and fixes the query is
    behaving exactly as intended and must not be driven to stop exploring.
    """
    del truth
    wasted = (
        len(trajectory.failed_queries)
        + len(trajectory.degenerate_queries)
        + len(trajectory.redundant_schema_inspections)
    )
    return min(wasted, MAX_PENALISED_ERRORS) / MAX_PENALISED_ERRORS


def efficiency_penalty(trajectory: Trajectory, truth: GroundTruth) -> float:
    """Fraction of the efficiency cap incurred, in ``[0, 1]``."""
    del truth
    excess = max(0, trajectory.n_calls - FREE_TOOL_CALLS)
    return min(excess, MAX_PENALISED_EXCESS_CALLS) / MAX_PENALISED_EXCESS_CALLS


def no_query_attempt_penalty(trajectory: Trajectory, truth: GroundTruth) -> float:
    """1.0 when the agent never even called ``execute_sql``.

    Deliberately keyed on *attempts*, not successes. If it fired whenever no
    query succeeded, it would stack with the error penalty and make an agent
    that tried five times and failed score worse than one that never tried --
    exactly the incentive the anti-hacking checklist warns about. Keyed this
    way, five failed attempts cost 0.15 while doing nothing costs 0.20, so
    attempting is always the better move.
    """
    del truth
    return 0.0 if trajectory.query_attempts else 1.0


# --------------------------------------------------------------------------
# metrics (logged, zero weight)
# --------------------------------------------------------------------------


def answered(trajectory: Trajectory, truth: GroundTruth) -> float:
    del truth
    return 1.0 if trajectory.termination is TerminationReason.SUBMITTED else 0.0


def n_tool_calls(trajectory: Trajectory, truth: GroundTruth) -> float:
    del truth
    return float(trajectory.n_calls)


def n_distinct_tables_read(trajectory: Trajectory, truth: GroundTruth) -> float:
    del truth
    return float(len(trajectory.tables_read))
