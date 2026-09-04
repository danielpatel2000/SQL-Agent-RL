"""Environment loop tests, driven by scripted agents -- no LLM involved."""

from __future__ import annotations

from pathlib import Path

import pytest

from sql_agent_rl.data import TEMPLATES_BY_ID, load_chinook
from sql_agent_rl.env import (
    SQLAgentEpisode,
    TerminationReason,
    ToolName,
)

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
        frames, {"year": 2023, "k": 3}
    )
    assert built is not None
    return built


@pytest.fixture()
def episode(task, chinook_db: Path):
    with SQLAgentEpisode(task, db_path=chinook_db) as ep:
        ep.reset()
        yield ep


def test_reset_returns_prompt_with_question_and_format(episode, task) -> None:
    prompt = episode.reset()
    assert task.question in prompt
    assert task.answer_format in prompt
    assert "inspect_schema" in episode.system_prompt
    assert "READ-ONLY" in episode.system_prompt


def test_happy_path_records_full_trajectory(episode) -> None:
    episode.inspect_schema()
    episode.execute_sql(TOP3_SQL)
    episode.submit_answer("1. Hugh O'Reilly\n2. Robert Brown\n3. Daan Peeters")

    traj = episode.trajectory
    assert episode.done
    assert traj.termination is TerminationReason.SUBMITTED
    assert traj.n_calls == 3
    assert len(traj.schema_inspections) == 1
    assert len(traj.grounded_queries) == 1
    assert traj.inspected_schema_before_first_query
    assert traj.tables_read == {"Customer", "Invoice", "InvoiceLine"}
    assert traj.submitted_answer.startswith("1. Hugh")
    assert [str(c.tool) for c in traj.calls] == [
        "inspect_schema",
        "execute_sql",
        "submit_answer",
    ]


def test_sql_errors_are_observations_not_crashes(episode) -> None:
    obs = episode.execute_sql("SELECT * FROM customers")
    assert "SQL ERROR" in obs
    assert not episode.done
    assert len(episode.trajectory.failed_queries) == 1
    # The agent can recover on the next call.
    assert "SQL ERROR" not in episode.execute_sql("SELECT COUNT(*) FROM Customer")


def test_destructive_sql_is_refused_without_ending_the_episode(episode) -> None:
    obs = episode.execute_sql("DROP TABLE Customer")
    assert "SQL ERROR" in obs and "read-only" in obs.lower()
    assert not episode.done


def test_budget_exhaustion_terminates_gracefully(task, chinook_db: Path) -> None:
    with SQLAgentEpisode(task, db_path=chinook_db, max_tool_calls=4) as ep:
        ep.reset()
        for _ in range(4):
            result = ep.step(ToolName.EXECUTE_SQL, query="SELECT COUNT(*) FROM Customer")
        assert result.done
        assert ep.done
        assert ep.trajectory.termination is TerminationReason.BUDGET_EXHAUSTED
        assert ep.trajectory.submitted_answer is None
        assert "budget" in result.observation.lower()


def test_calls_after_termination_are_refused_not_crashes(episode) -> None:
    episode.submit_answer("anything")
    assert episode.done
    result = episode.step(ToolName.EXECUTE_SQL, query="SELECT 1")
    assert result.done
    assert "already ended" in result.observation
    # The refused call must not be appended to the trajectory.
    assert episode.trajectory.n_calls == 1


def test_budget_warning_appears_near_the_end(task, chinook_db: Path) -> None:
    with SQLAgentEpisode(task, db_path=chinook_db, max_tool_calls=5) as ep:
        ep.reset()
        assert "remaining" not in ep.inspect_schema()
        obs = ep.execute_sql("SELECT COUNT(*) FROM Customer")
        assert "3 tool calls remaining" in obs


def test_unknown_tool_burns_budget_without_being_miscounted(episode) -> None:
    """Regression: a bogus tool name must not count as a schema inspection."""
    result = episode.step("list_tables")
    assert not result.done
    assert "Unknown tool" in result.observation
    traj = episode.trajectory
    assert traj.n_calls == 1
    assert traj.schema_inspections == []
    assert not traj.inspected_schema_before_first_query
    assert str(traj.calls[0].tool) == "unknown"


def test_missing_query_argument_is_an_observation(episode) -> None:
    result = episode.step(ToolName.EXECUTE_SQL)
    assert not result.done
    assert "requires a `query`" in result.observation


def test_schema_after_first_query_does_not_count_as_schema_first(episode) -> None:
    episode.execute_sql("SELECT COUNT(*) FROM Customer")
    episode.inspect_schema()
    assert not episode.trajectory.inspected_schema_before_first_query


def test_degenerate_and_metadata_queries_are_not_grounded(episode) -> None:
    episode.execute_sql("SELECT 'Acme Corp' AS name")
    episode.execute_sql("SELECT name FROM sqlite_master")
    traj = episode.trajectory
    assert len(traj.successful_queries) == 2
    assert traj.grounded_queries == []
    assert len(traj.degenerate_queries) == 2


def test_truncation_is_reported_to_the_agent(task, chinook_db: Path) -> None:
    with SQLAgentEpisode(task, db_path=chinook_db, max_rows=5) as ep:
        ep.reset()
        obs = ep.execute_sql("SELECT TrackId FROM Track")
        assert "TRUNCATED" in obs
        assert ep.trajectory.calls[0].query_result.truncated


def test_trajectory_log_dict_is_json_serialisable(episode) -> None:
    import json

    episode.inspect_schema()
    episode.execute_sql(TOP3_SQL)
    episode.submit_answer("1. Hugh O'Reilly")
    payload = json.loads(json.dumps(episode.trajectory.to_log_dict()))
    assert payload["n_grounded_queries"] == 1
    assert payload["termination"] == "submitted"
    assert payload["tables_read"] == ["Customer", "Invoice", "InvoiceLine"]


def test_episode_can_run_every_template(chinook_db: Path) -> None:
    """A scripted agent drives one episode per template without crashing."""
    from sql_agent_rl.data import generate_tasks

    frames = load_chinook(chinook_db)
    for t in generate_tasks(frames, 20, seed=42):
        with SQLAgentEpisode(t, db_path=chinook_db) as ep:
            ep.reset()
            ep.inspect_schema()
            ep.execute_sql("SELECT COUNT(*) FROM Invoice")
            ep.submit_answer("placeholder")
            assert ep.done
            assert ep.trajectory.termination is TerminationReason.SUBMITTED
