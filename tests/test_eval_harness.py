"""End-to-end test of the eval harness, driven by a stubbed model client.

``eval/run_eval.py`` normally talks to a live OpenAI-compatible endpoint. Here
the client is replaced by a scripted stub that emits the same response shape,
so the whole loop -- tool-call dispatch, observation feedback, termination,
reward breakdown -- is exercised in CI without an API key or a network call.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from sql_agent_rl.data import TEMPLATES_BY_ID, load_chinook

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_run_eval():
    """Import ``eval/run_eval.py``, which is a script rather than a package.

    The module must be registered in ``sys.modules`` *before* execution:
    ``@dataclass`` resolves annotations by looking its own module up there, and
    fails with an opaque AttributeError if it is absent.
    """
    spec = importlib.util.spec_from_file_location(
        "run_eval", REPO_ROOT / "eval" / "run_eval.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["run_eval"] = module
    spec.loader.exec_module(module)
    return module


run_eval = _load_run_eval()

TOP3_SQL = (
    "SELECT c.FirstName, c.LastName, ROUND(SUM(il.UnitPrice * il.Quantity), 2) AS rev "
    "FROM Customer c JOIN Invoice i ON i.CustomerId = c.CustomerId "
    "JOIN InvoiceLine il ON il.InvoiceId = i.InvoiceId "
    "WHERE CAST(strftime('%Y', i.InvoiceDate) AS INT) = 2023 "
    "GROUP BY c.CustomerId ORDER BY rev DESC LIMIT 3"
)


def _tool_call(call_id: str, name: str, **args) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(args)),
    )


class StubClient:
    """Mimics ``AsyncOpenAI`` closely enough for the eval loop.

    ``turns`` is a list of tool-call batches; each ``create`` call returns the
    next one. A ``None`` entry produces a plain assistant message with no tool
    calls, which exercises the "nudge the model back to tools" branch.
    """

    def __init__(self, turns: list, fail_with: Exception | None = None) -> None:
        self.turns = list(turns)
        self.fail_with = fail_with
        self.requests: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.requests.append(kwargs)
        if self.fail_with is not None:
            raise self.fail_with
        calls = self.turns.pop(0) if self.turns else None
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=None if calls else "thinking out loud",
                        tool_calls=calls,
                    )
                )
            ]
        )


@pytest.fixture()
def task(chinook_db: Path):
    frames = load_chinook(chinook_db)
    built = TEMPLATES_BY_ID["top_customers_by_revenue"].build(
        frames, {"year": 2023, "k": 3}
    )
    assert built is not None
    return built


def _run(client, task, chinook_db: Path, max_tool_calls: int = 12):
    return asyncio.run(
        run_eval.run_episode(
            client, "stub-model", task, chinook_db, max_tool_calls, 0.0
        )
    )


def test_competent_model_run_scores_full_reward(task, chinook_db: Path) -> None:
    client = StubClient(
        [
            [_tool_call("1", "inspect_schema")],
            [_tool_call("2", "execute_sql", query=TOP3_SQL)],
            [
                _tool_call(
                    "3",
                    "submit_answer",
                    answer="1. Hugh O'Reilly\n2. Robert Brown\n3. Daan Peeters",
                )
            ],
        ]
    )
    outcome = _run(client, task, chinook_db)
    assert outcome.error is None
    assert outcome.breakdown.total == pytest.approx(1.05)
    assert outcome.trajectory_log["n_grounded_queries"] == 1
    assert outcome.trajectory_log["termination"] == "submitted"

    # The model really was given the tools and the system prompt.
    first = client.requests[0]
    assert {t["function"]["name"] for t in first["tools"]} == {
        "inspect_schema",
        "execute_sql",
        "submit_answer",
    }
    assert first["messages"][0]["role"] == "system"
    assert "READ-ONLY" in first["messages"][0]["content"]


def test_tool_observations_are_fed_back_to_the_model(task, chinook_db: Path) -> None:
    client = StubClient(
        [
            [_tool_call("1", "execute_sql", query="SELECT * FROM customers")],
            [_tool_call("2", "execute_sql", query=TOP3_SQL)],
            [_tool_call("3", "submit_answer", answer="1. Hugh O'Reilly")],
        ]
    )
    _run(client, task, chinook_db)
    # By the third request the transcript must contain the error observation,
    # which is what lets a model self-correct.
    transcript = client.requests[-1]["messages"]
    tool_messages = [m for m in transcript if m["role"] == "tool"]
    assert any("SQL ERROR" in m["content"] for m in tool_messages)
    assert any("Hugh" in m["content"] for m in tool_messages)


def test_model_that_never_calls_a_tool_is_nudged_then_ends(
    task, chinook_db: Path
) -> None:
    client = StubClient([None, None, None])
    outcome = _run(client, task, chinook_db, max_tool_calls=3)
    assert outcome.error is None
    assert outcome.trajectory_log["n_calls"] == 0
    # No answer, no query: scored, not crashed.
    assert outcome.breakdown.raw["answered"] == 0.0
    assert outcome.breakdown.total < 0


def test_api_failure_is_reported_not_raised(task, chinook_db: Path) -> None:
    client = StubClient([], fail_with=RuntimeError("upstream 503"))
    outcome = _run(client, task, chinook_db)
    assert outcome.error is not None
    assert "upstream 503" in outcome.error
    # A failed episode still yields a scored trajectory so the sweep continues.
    assert outcome.breakdown.total <= 0


def test_malformed_tool_arguments_do_not_crash(task, chinook_db: Path) -> None:
    bad = SimpleNamespace(
        id="1",
        function=SimpleNamespace(name="execute_sql", arguments="{not json"),
    )
    client = StubClient([[bad], [_tool_call("2", "submit_answer", answer="x")]])
    outcome = _run(client, task, chinook_db)
    assert outcome.error is None
    assert outcome.trajectory_log["n_calls"] == 2


def test_hallucinated_tool_name_is_handled(task, chinook_db: Path) -> None:
    client = StubClient(
        [
            [_tool_call("1", "list_tables")],
            [_tool_call("2", "submit_answer", answer="x")],
        ]
    )
    outcome = _run(client, task, chinook_db)
    assert outcome.error is None
    assert outcome.trajectory_log["calls"][0]["tool"] == "unknown"


def test_budget_is_enforced_against_a_looping_model(task, chinook_db: Path) -> None:
    client = StubClient(
        [[_tool_call(str(i), "inspect_schema")] for i in range(20)]
    )
    outcome = _run(client, task, chinook_db, max_tool_calls=5)
    assert outcome.trajectory_log["n_calls"] == 5
    assert outcome.trajectory_log["termination"] == "budget_exhausted"


def test_summarise_runs_over_mixed_outcomes(task, chinook_db: Path, capsys) -> None:
    good = _run(
        StubClient(
            [
                [_tool_call("1", "inspect_schema")],
                [_tool_call("2", "execute_sql", query=TOP3_SQL)],
                [
                    _tool_call(
                        "3", "submit_answer",
                        answer="1. Hugh O'Reilly\n2. Robert Brown\n3. Daan Peeters",
                    )
                ],
            ]
        ),
        task,
        chinook_db,
    )
    bad = _run(StubClient([[_tool_call("1", "submit_answer", answer="?")]]), task, chinook_db)

    run_eval.summarise([good, bad])
    out = capsys.readouterr().out
    assert "EPISODES: 2" in out
    assert "fully correct          1/2" in out
    assert "grounding gate fired   1/2" in out
    for name in ("correctness", "grounding", "wasted_call_penalty"):
        assert name in out
    assert "top_customers_by_revenue" in out


def test_summarise_handles_an_empty_run(capsys) -> None:
    run_eval.summarise([])
    assert "No episodes completed." in capsys.readouterr().out
