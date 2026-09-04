"""Tests for the ``verifiers`` adapter.

These check the *translation layer* only -- that tool schemas, state handling
and rubric weights line up with the framework-independent core. The end-to-end
run against a real model is ``eval/run_eval.py``; it needs an API key and so is
not part of the unit suite.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from sql_agent_rl.rubric import REWARD_COMPONENTS, compute_reward

vf = pytest.importorskip("verifiers", reason="verifiers extra not installed")
pytest.importorskip("datasets", reason="datasets not installed")

from sql_agent_rl.env.verifiers_env import (  # noqa: E402
    EPISODE_KEY,
    SQLAgentEnv,
    build_rubric,
    load_environment,
)

TOP3_SQL = """
SELECT c.FirstName, c.LastName, ROUND(SUM(il.UnitPrice * il.Quantity), 2) AS rev
FROM Customer c
JOIN Invoice i ON i.CustomerId = c.CustomerId
JOIN InvoiceLine il ON il.InvoiceId = i.InvoiceId
WHERE CAST(strftime('%Y', i.InvoiceDate) AS INT) = 2023
GROUP BY c.CustomerId ORDER BY rev DESC LIMIT 3
"""


@pytest.fixture(scope="module")
def env(chinook_db: Path) -> SQLAgentEnv:
    return load_environment(db_path=chinook_db, num_tasks=8, seed=0)


@pytest.fixture(scope="module")
def ranked_env(chinook_db: Path) -> SQLAgentEnv:
    """An env restricted to one template, so the reward scripts below apply."""
    return load_environment(
        db_path=chinook_db,
        num_tasks=4,
        seed=0,
        template_ids=["top_customers_by_revenue"],
    )


def test_environment_builds_with_the_three_tools(env: SQLAgentEnv) -> None:
    assert isinstance(env, vf.StatefulToolEnv)
    assert set(env.tool_map) == {"inspect_schema", "execute_sql", "submit_answer"}
    assert len(env.dataset) == 8


def test_episode_handle_is_hidden_from_the_model_schema(env: SQLAgentEnv) -> None:
    """The model must never see (or be able to forge) the episode argument."""
    for tool_def in env.tool_defs:
        properties = tool_def.parameters.get("properties", {})
        assert "episode" not in properties, tool_def.name
        assert "episode" not in tool_def.parameters.get("required", [])
    assert env.skipped_args["execute_sql"] == ["episode"]


def test_dataset_rows_carry_what_the_rollout_needs(env: SQLAgentEnv) -> None:
    row = env.dataset[0]
    assert row["info"]["task_id"] in env.tasks_by_id
    assert "Answer format:" in row["question"]


def test_setup_state_creates_a_fresh_episode(env: SQLAgentEnv) -> None:
    state = {"info": dict(env.dataset[0]["info"])}
    state = asyncio.run(env.setup_state(state))
    episode = state[EPISODE_KEY]
    assert episode.trajectory.n_calls == 0
    assert not episode.done

    other = asyncio.run(env.setup_state({"info": dict(env.dataset[0]["info"])}))
    assert other[EPISODE_KEY] is not episode, "episodes must not be shared"


def test_tools_drive_the_episode_and_update_the_trajectory(env: SQLAgentEnv) -> None:
    state = asyncio.run(env.setup_state({"info": dict(env.dataset[0]["info"])}))
    episode = state[EPISODE_KEY]

    args = env.update_tool_args("inspect_schema", {}, [], state)
    assert args["episode"] is episode
    assert "TABLE Customer" in env.inspect_schema(**args)

    args = env.update_tool_args("execute_sql", {"query": "SELECT 1"}, [], state)
    env.execute_sql(**args)
    assert len(episode.trajectory.query_attempts) == 1


def test_episode_finished_stop_condition_follows_the_episode(env: SQLAgentEnv) -> None:
    state = asyncio.run(env.setup_state({"info": dict(env.dataset[0]["info"])}))
    episode = state[EPISODE_KEY]
    assert not asyncio.run(env.episode_finished(state))
    episode.submit_answer("done")
    assert asyncio.run(env.episode_finished(state))


def test_episode_finished_is_registered_alongside_the_base_stop_conditions(
    env: SQLAgentEnv,
) -> None:
    """It must compose with the base stop conditions, not replace them."""
    assert getattr(type(env).episode_finished, "stop", False) is True
    registered = {c.__name__ for c in env._stop_conditions}
    assert "episode_finished" in registered
    # The base class's own conditions must still be active.
    assert {"has_error", "max_turns_reached"} <= registered


def test_rubric_weights_match_the_core_rubric() -> None:
    rubric = build_rubric()
    names = [f.__name__ for f in rubric.funcs]
    weights = dict(zip(names, rubric.weights))
    for component in REWARD_COMPONENTS:
        assert component.name in weights, f"{component.name} missing from rubric"
        expected = 0.0 if (component.is_metric or component.is_gate) else component.weight
        assert weights[component.name] == pytest.approx(expected)


@pytest.mark.parametrize(
    "script_name",
    ["ideal", "laundered", "no_query", "self_corrected"],
)
def test_verifiers_rubric_total_matches_compute_reward(
    ranked_env: SQLAgentEnv, script_name: str
) -> None:
    """The adapter must not change the reward, only how it is reported."""
    task_id, task = next(iter(ranked_env.tasks_by_id.items()))
    state = asyncio.run(ranked_env.setup_state({"info": {"task_id": task_id}}))
    episode = state[EPISODE_KEY]
    answer = "1. Hugh O'Reilly\n2. Robert Brown\n3. Daan Peeters"
    sql = TOP3_SQL.replace(
        "= 2023", f"= {task.params['year']}"
    ).replace("LIMIT 3", f"LIMIT {task.params['k']}")

    if script_name == "ideal":
        episode.inspect_schema()
        episode.execute_sql(sql)
        episode.submit_answer(answer)
    elif script_name == "laundered":
        episode.execute_sql("SELECT 'Hugh O''Reilly' AS a")
        episode.submit_answer(answer)
    elif script_name == "no_query":
        episode.submit_answer(answer)
    else:
        episode.inspect_schema()
        episode.execute_sql("SELECT * FROM customers")
        episode.execute_sql(sql)
        episode.submit_answer(answer)

    expected = compute_reward(episode.trajectory).total

    rubric = build_rubric()
    reward_funcs = [
        (f, w) for f, w in zip(rubric.funcs, rubric.weights)
    ]

    async def total() -> float:
        out = 0.0
        for fn, weight in reward_funcs:
            out += weight * await fn(state=state)
        return out

    assert asyncio.run(total()) == pytest.approx(expected), script_name


def test_rubric_returns_zero_without_an_episode() -> None:
    """A malformed state must not raise inside a rollout."""
    rubric = build_rubric()

    async def run() -> list[float]:
        return [await fn(state={}) for fn in rubric.funcs]

    assert all(v == 0.0 for v in asyncio.run(run()))


def test_missing_database_raises_a_helpful_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="download_chinook"):
        load_environment(db_path=tmp_path / "absent.db")
