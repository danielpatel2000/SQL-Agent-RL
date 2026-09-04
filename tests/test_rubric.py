"""Reward rubric tests, including the adversarial trajectories.

Every test here builds a trajectory by hand or with a scripted agent and scores
it. No LLM, and (for the hand-built ones) no database either -- which is the
point of making each component a pure function of
``(Trajectory, GroundTruth)``.

The adversarial cases come straight from the project's anti-reward-hacking
checklist: each one is an episode that *looks* completed, and each must score
low.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sql_agent_rl.data import TEMPLATES_BY_ID, load_chinook
from sql_agent_rl.data.answers import AnswerType, Entity, GroundTruth
from sql_agent_rl.env import SQLAgentEpisode, TerminationReason, ToolName, Trajectory
from sql_agent_rl.env.trajectory import ToolCallRecord
from sql_agent_rl.rubric import (
    MAX_TOTAL_REWARD,
    MIN_TOTAL_REWARD,
    REWARD_COMPONENTS,
    compute_reward,
)
from sql_agent_rl.rubric import components as C
from sql_agent_rl.sandbox.executor import QueryResult

YEAR = 2023
K = 3
CORRECT = "1. Hugh O'Reilly\n2. Robert Brown\n3. Daan Peeters"
TOP3_SQL = """
SELECT c.FirstName, c.LastName, ROUND(SUM(il.UnitPrice * il.Quantity), 2) AS rev
FROM Customer c
JOIN Invoice i ON i.CustomerId = c.CustomerId
JOIN InvoiceLine il ON il.InvoiceId = i.InvoiceId
WHERE CAST(strftime('%Y', i.InvoiceDate) AS INT) = 2023
GROUP BY c.CustomerId ORDER BY rev DESC LIMIT 3
"""


@pytest.fixture()
def task(chinook_db: Path):
    frames = load_chinook(chinook_db)
    built = TEMPLATES_BY_ID["top_customers_by_revenue"].build(
        frames, {"year": YEAR, "k": K}
    )
    assert built is not None
    return built


def run(task, chinook_db: Path, script) -> Trajectory:
    """Drive one episode with a scripted agent and return its trajectory."""
    with SQLAgentEpisode(task, db_path=chinook_db, max_tool_calls=12) as ep:
        ep.reset()
        script(ep)
        return ep.trajectory


# --------------------------------------------------------------------------
# baseline: the behaviour we actually want
# --------------------------------------------------------------------------


def test_ideal_episode_scores_near_maximum(task, chinook_db: Path) -> None:
    def script(ep):
        ep.inspect_schema()
        ep.execute_sql(TOP3_SQL)
        ep.submit_answer(CORRECT)

    breakdown = compute_reward(run(task, chinook_db, script))
    assert breakdown.total == pytest.approx(1.05)
    assert not breakdown.gated
    assert breakdown.raw["correctness"] == 1.0
    assert breakdown.raw["grounding"] == 1.0
    assert breakdown.raw["schema_first_bonus"] == 1.0


def test_self_correction_stays_clearly_worthwhile(task, chinook_db: Path) -> None:
    """Two errors then a fix must still beat almost everything else."""

    def script(ep):
        ep.inspect_schema()
        ep.execute_sql("SELECT * FROM customers")  # wrong table name
        ep.execute_sql("SELECT nope FROM Customer")  # wrong column
        ep.execute_sql(TOP3_SQL)
        ep.submit_answer(CORRECT)

    breakdown = compute_reward(run(task, chinook_db, script))
    assert breakdown.total == pytest.approx(1.05 - 0.06)
    assert breakdown.raw["correctness"] == 1.0
    # Error pressure must stay small relative to correctness.
    assert abs(breakdown.weighted["wasted_call_penalty"]) < 0.1


# --------------------------------------------------------------------------
# adversarial: hardcoded / literal answer laundering
# --------------------------------------------------------------------------


def test_literal_laundering_is_gated_to_zero(task, chinook_db: Path) -> None:
    """`SELECT 'Hugh O''Reilly'` executes fine but reads no data table."""

    def script(ep):
        ep.execute_sql(
            "SELECT 'Hugh O''Reilly' AS a, 'Robert Brown' AS b, 'Daan Peeters' AS c"
        )
        ep.submit_answer(CORRECT)

    breakdown = compute_reward(run(task, chinook_db, script))
    assert breakdown.gated
    assert breakdown.weighted["correctness"] == 0.0
    assert breakdown.total < 0
    # The answer itself was right -- only grounding stopped it paying out.
    assert breakdown.raw["correctness"] == 1.0
    assert breakdown.raw["grounding"] == 0.0


def test_catalogue_browsing_does_not_satisfy_grounding(task, chinook_db: Path) -> None:
    """Reading sqlite_master is metadata, not data."""

    def script(ep):
        ep.execute_sql("SELECT name FROM sqlite_master WHERE type = 'table'")
        ep.submit_answer(CORRECT)

    breakdown = compute_reward(run(task, chinook_db, script))
    assert breakdown.gated
    assert breakdown.total < 0


def test_trivial_select_one_does_not_satisfy_grounding(task, chinook_db: Path) -> None:
    def script(ep):
        ep.execute_sql("SELECT 1")
        ep.submit_answer(CORRECT)

    assert compute_reward(run(task, chinook_db, script)).gated


def test_real_query_on_an_unrelated_table_still_needs_a_traceable_answer(
    task, chinook_db: Path
) -> None:
    """Querying *something* real is not enough; the answer must appear in it."""

    def script(ep):
        ep.execute_sql("SELECT Name FROM Genre LIMIT 5")
        ep.submit_answer(CORRECT)

    breakdown = compute_reward(run(task, chinook_db, script))
    assert breakdown.raw["grounding"] == 0.0
    assert breakdown.gated


# --------------------------------------------------------------------------
# adversarial: schema-inspection farming
# --------------------------------------------------------------------------


def test_schema_farming_earns_the_bonus_only_once(task, chinook_db: Path) -> None:
    def script(ep):
        for _ in range(9):
            ep.inspect_schema()

    breakdown = compute_reward(run(task, chinook_db, script))
    assert breakdown.raw["schema_first_bonus"] == 1.0
    assert breakdown.weighted["schema_first_bonus"] == pytest.approx(0.05)
    # Farming is strictly loss-making once efficiency and no-query bite.
    assert breakdown.total < 0


def test_farming_is_worse_than_a_single_inspection(task, chinook_db: Path) -> None:
    def once(ep):
        ep.inspect_schema()
        ep.execute_sql(TOP3_SQL)
        ep.submit_answer(CORRECT)

    def many(ep):
        for _ in range(6):
            ep.inspect_schema()
        ep.execute_sql(TOP3_SQL)
        ep.submit_answer(CORRECT)

    assert (
        compute_reward(run(task, chinook_db, many)).total
        < compute_reward(run(task, chinook_db, once)).total
    )


# --------------------------------------------------------------------------
# adversarial: cherry-picked / truncated results
# --------------------------------------------------------------------------


def test_answer_matching_a_truncated_result_is_still_graded_in_full(
    task, chinook_db: Path
) -> None:
    """Grade against recomputed ground truth, never "did the rows look right"."""

    def script(ep):
        # A deliberately truncated view: only the top 2 of the required 3.
        ep.execute_sql(TOP3_SQL.replace("LIMIT 3", "LIMIT 2"))
        ep.submit_answer("1. Hugh O'Reilly\n2. Robert Brown")

    breakdown = compute_reward(run(task, chinook_db, script))
    # Grounded (the rows were real) but only 2 of 3 positions are filled.
    assert not breakdown.gated
    assert breakdown.raw["correctness"] == pytest.approx(2 / 3)
    assert breakdown.total < 1.0


def test_row_limit_truncation_cannot_launder_a_wrong_answer(
    task, chinook_db: Path
) -> None:
    def script(ep):
        ep.execute_sql("SELECT FirstName, LastName FROM Customer")
        ep.submit_answer("1. Luis Goncalves\n2. Leonie Kohler\n3. Francois Tremblay")

    breakdown = compute_reward(run(task, chinook_db, script))
    # It is grounded -- those names really were retrieved -- but simply wrong.
    assert breakdown.raw["grounding"] == 1.0
    assert breakdown.raw["correctness"] == 0.0
    assert breakdown.total <= 0.0


def test_dumping_every_customer_does_not_score(task, chinook_db: Path) -> None:
    def script(ep):
        ep.execute_sql("SELECT FirstName, LastName FROM Customer")
        rows = ep.trajectory.calls[-1].query_result.rows
        ep.submit_answer("\n".join(f"{a} {b}" for a, b in rows))

    breakdown = compute_reward(run(task, chinook_db, script))
    # Ranked answers truncate to K, so a dump fills the first K positions with
    # whatever happened to sort first -- not the right answer.
    assert breakdown.raw["correctness"] < 0.5


# --------------------------------------------------------------------------
# adversarial: error-spam gaming
# --------------------------------------------------------------------------


def test_never_querying_scores_worse_than_querying_and_failing(
    task, chinook_db: Path
) -> None:
    """The invariant that keeps exploration rational."""

    def never(ep):
        ep.submit_answer(CORRECT)

    def tried_and_failed(ep):
        for _ in range(5):
            ep.execute_sql("SELECT * FROM customers")
        ep.submit_answer(CORRECT)

    never_reward = compute_reward(run(task, chinook_db, never)).total
    failed_reward = compute_reward(run(task, chinook_db, tried_and_failed)).total
    assert failed_reward > never_reward, (
        f"attempting and failing ({failed_reward:+.3f}) must beat never "
        f"attempting ({never_reward:+.3f})"
    )


def test_error_penalty_is_capped(task, chinook_db: Path) -> None:
    def script(ep):
        for _ in range(10):
            ep.execute_sql("SELECT * FROM customers")

    breakdown = compute_reward(run(task, chinook_db, script))
    assert breakdown.weighted["wasted_call_penalty"] == pytest.approx(-0.15)


def test_degenerate_queries_are_charged_like_errors(task, chinook_db: Path) -> None:
    """Closes the "burn calls on SELECT 1 for free" loophole."""

    def script(ep):
        for _ in range(3):
            ep.execute_sql("SELECT 1")
        ep.submit_answer(CORRECT)

    breakdown = compute_reward(run(task, chinook_db, script))
    assert breakdown.weighted["wasted_call_penalty"] == pytest.approx(-0.09)


def test_no_query_penalty_does_not_stack_with_error_penalty(
    task, chinook_db: Path
) -> None:
    """If it stacked, failing five times would be worse than doing nothing."""

    def script(ep):
        for _ in range(5):
            ep.execute_sql("SELECT * FROM customers")

    breakdown = compute_reward(run(task, chinook_db, script))
    assert breakdown.raw["no_query_attempt_penalty"] == 0.0


# --------------------------------------------------------------------------
# budget behaviour
# --------------------------------------------------------------------------


def test_budget_exhaustion_yields_no_correctness_and_does_not_crash(
    task, chinook_db: Path
) -> None:
    with SQLAgentEpisode(task, db_path=chinook_db, max_tool_calls=3) as ep:
        ep.reset()
        for _ in range(5):
            ep.execute_sql("SELECT COUNT(*) FROM Customer")
    breakdown = compute_reward(ep.trajectory)
    assert ep.trajectory.termination is TerminationReason.BUDGET_EXHAUSTED
    assert breakdown.raw["correctness"] == 0.0
    assert breakdown.raw["answered"] == 0.0
    assert MIN_TOTAL_REWARD <= breakdown.total <= MAX_TOTAL_REWARD


def test_efficiency_penalty_only_bites_past_the_free_budget(
    task, chinook_db: Path
) -> None:
    def short(ep):
        ep.inspect_schema()
        ep.execute_sql(TOP3_SQL)
        ep.submit_answer(CORRECT)

    breakdown = compute_reward(run(task, chinook_db, short))
    assert breakdown.weighted["efficiency_penalty"] == 0.0


# --------------------------------------------------------------------------
# structural guarantees about the rubric itself
# --------------------------------------------------------------------------


def test_reward_stays_within_documented_bounds(task, chinook_db: Path) -> None:
    from sql_agent_rl.data import generate_tasks

    frames = load_chinook(chinook_db)
    scripts = [
        lambda ep: ep.submit_answer("nothing"),
        lambda ep: [ep.inspect_schema() for _ in range(12)],
        lambda ep: [ep.execute_sql("SELECT * FROM nope") for _ in range(12)],
        lambda ep: (ep.inspect_schema(), ep.execute_sql(TOP3_SQL), ep.submit_answer(CORRECT)),
    ]
    for t in generate_tasks(frames, 6, seed=1):
        for script in scripts:
            breakdown = compute_reward(run(t, chinook_db, script))
            assert MIN_TOTAL_REWARD <= breakdown.total <= MAX_TOTAL_REWARD


def test_process_shaping_cannot_outweigh_correctness() -> None:
    """The 'nudges, not the point' claim, asserted rather than asserted-in-prose."""
    positive_shaping = sum(
        c.weight
        for c in REWARD_COMPONENTS
        if not c.is_metric and not c.is_gate and c.weight > 0 and c.name != "correctness"
    )
    correctness_weight = next(
        c.weight for c in REWARD_COMPONENTS if c.name == "correctness"
    )
    assert positive_shaping <= 0.05 * correctness_weight


def test_every_component_is_a_pure_function_of_trajectory_and_truth() -> None:
    """Hand-built trajectory, no database, no episode, no model."""
    truth = GroundTruth(
        answer_type=AnswerType.RANKED_LIST,
        tie_groups=((Entity.make(1, "Alice"),), (Entity.make(2, "Bob"),)),
        k=2,
    )
    result = QueryResult(
        sql="SELECT ...",
        ok=True,
        columns=("name",),
        rows=(("Alice",), ("Bob",)),
        row_count=2,
        tables_read=frozenset({"Customer"}),
    )
    traj = Trajectory(task=_FakeTask(truth), max_tool_calls=12)
    traj.calls = [
        ToolCallRecord(0, ToolName.INSPECT_SCHEMA, {}, True, "schema"),
        ToolCallRecord(1, ToolName.EXECUTE_SQL, {"query": "..."}, True, "rows", result),
        ToolCallRecord(2, ToolName.SUBMIT_ANSWER, {"answer": "x"}, True, "ok"),
    ]
    traj.submitted_answer = "1. Alice\n2. Bob"
    traj.termination = TerminationReason.SUBMITTED

    for component in REWARD_COMPONENTS:
        first = component.fn(traj, truth)
        assert first == component.fn(traj, truth), f"{component.name} is not pure"

    assert C.correctness(traj, truth) == 1.0
    assert C.grounding(traj, truth) == 1.0
    assert compute_reward(traj, truth).total == pytest.approx(1.05)


class _FakeTask:
    """Minimal stand-in so trajectories can be built with no database."""

    template_id = "fake"
    difficulty = "single_table"
    params: dict = {}

    def __init__(self, truth: GroundTruth) -> None:
        self.ground_truth = truth


def test_breakdown_is_logged_per_component(task, chinook_db: Path) -> None:
    def script(ep):
        ep.inspect_schema()
        ep.execute_sql(TOP3_SQL)
        ep.submit_answer(CORRECT)

    breakdown = compute_reward(run(task, chinook_db, script))
    log = breakdown.to_log_dict()
    for component in REWARD_COMPONENTS:
        assert f"raw/{component.name}" in log
        assert f"weighted/{component.name}" in log
    assert log["reward"] == breakdown.total
    assert "correctness=+1.000" in str(breakdown)
