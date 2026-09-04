"""The multi-turn episode: three tools, a call budget, and a trajectory log.

This is deliberately framework-independent -- no ``verifiers``, no Gymnasium,
no LLM. It is driveable by a scripted agent (see ``tests/test_episode.py``),
which is what makes the environment testable on its own and portable to
another harness later. The ``verifiers`` integration in
:mod:`sql_agent_rl.env.verifiers_env` is a thin adapter over this class.

Separation of mechanics and reward
----------------------------------
The episode never refuses a legal-but-unwise action. ``submit_answer`` is
accepted even when the agent has run no query at all: policing that here would
hide the behaviour from the reward function, and the whole point of the project
is to *shape* it. So the environment records what happened and the rubric
decides what it was worth. That also means adversarial trajectories can be
constructed and scored in tests, which they could not be if the environment
rejected them up front.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sql_agent_rl.data.tasks import Task
from sql_agent_rl.env.trajectory import (
    TerminationReason,
    ToolCallRecord,
    ToolName,
    Trajectory,
)
from sql_agent_rl.sandbox.executor import (
    DEFAULT_MAX_ROWS,
    DEFAULT_TIMEOUT_S,
    SQLSandbox,
)

DEFAULT_MAX_TOOL_CALLS = 12

SYSTEM_PROMPT = """\
You are a data analyst answering a question about a music store's relational \
database. You have three tools:

  inspect_schema()      -- returns the tables, columns, keys and row counts.
  execute_sql(query)    -- runs ONE read-only SQL statement and returns the rows.
  submit_answer(answer) -- submits your final answer and ends the episode.

Rules:
  * The database is READ-ONLY. Only SELECT, WITH and EXPLAIN statements run;
    anything else is rejected. One statement per call, no semicolon-chaining.
  * Results are truncated to {max_rows} rows. If you see a truncation notice,
    aggregate or filter in SQL rather than assuming the rows you got are the
    whole story.
  * You have a budget of {max_tool_calls} tool calls for the whole episode. If
    you run out before submitting, the episode ends with no answer.
  * Inspect the schema before writing your first query -- the column names are
    probably not what you would guess.
  * Base your answer on query results you actually retrieved, not on prior
    knowledge of this dataset. An answer that is not traceable to a query you
    ran scores zero even if it happens to be right.
"""


@dataclass(frozen=True)
class StepResult:
    """What the caller gets back from one tool call."""

    observation: str
    done: bool
    termination: TerminationReason
    record: ToolCallRecord | None = None


class SQLAgentEpisode:
    """One episode: a task, a sandboxed database, and a bounded tool loop."""

    def __init__(
        self,
        task: Task,
        db_path: str | Path | None = None,
        sandbox: SQLSandbox | None = None,
        max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
        max_rows: int = DEFAULT_MAX_ROWS,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        schema_sample_rows: int = 0,
    ) -> None:
        if sandbox is None:
            if db_path is None:
                raise ValueError("Provide either `sandbox` or `db_path`.")
            sandbox = SQLSandbox(db_path, max_rows=max_rows, timeout_s=timeout_s)
            self._owns_sandbox = True
        else:
            self._owns_sandbox = False
        self.task = task
        self.sandbox = sandbox
        self.max_tool_calls = max_tool_calls
        self.schema_sample_rows = schema_sample_rows
        self.trajectory = Trajectory(task=task, max_tool_calls=max_tool_calls)

    # -- lifecycle ------------------------------------------------------------

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT.format(
            max_rows=self.sandbox.max_rows, max_tool_calls=self.max_tool_calls
        )

    def reset(self) -> str:
        """Start the episode and return the initial user-facing prompt."""
        self.trajectory = Trajectory(
            task=self.task, max_tool_calls=self.max_tool_calls
        )
        return self.task.prompt

    @property
    def done(self) -> bool:
        return self.trajectory.termination is not TerminationReason.RUNNING

    @property
    def calls_remaining(self) -> int:
        return max(0, self.max_tool_calls - self.trajectory.n_calls)

    def close(self) -> None:
        if self._owns_sandbox:
            self.sandbox.close()

    def __enter__(self) -> SQLAgentEpisode:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- the tool loop --------------------------------------------------------

    def step(self, tool: ToolName | str, **kwargs: Any) -> StepResult:
        """Execute one tool call.

        Never raises for agent-caused problems -- an unknown tool, a bad
        argument or a call past the budget all come back as an explanatory
        observation, because a crash mid-rollout would lose the episode.
        """
        if self.done:
            return StepResult(
                observation=(
                    "This episode has already ended "
                    f"({self.trajectory.termination}). No further calls are accepted."
                ),
                done=True,
                termination=self.trajectory.termination,
            )

        requested = str(tool)
        try:
            tool = ToolName(requested)
        except ValueError:
            tool = ToolName.UNKNOWN
        if tool is ToolName.UNKNOWN:
            available = ", ".join(
                t.value for t in ToolName if t is not ToolName.UNKNOWN
            )
            return self._record(
                ToolName.UNKNOWN,
                {"requested_tool": requested},
                ok=False,
                observation=f"Unknown tool '{requested}'. Available tools: {available}.",
            )

        if tool is ToolName.INSPECT_SCHEMA:
            return self._inspect_schema()
        if tool is ToolName.EXECUTE_SQL:
            return self._execute_sql(kwargs.get("query"))
        return self._submit_answer(kwargs.get("answer"))

    # -- tools ----------------------------------------------------------------

    def _inspect_schema(self) -> StepResult:
        schema = self.sandbox.describe_schema(sample_rows=self.schema_sample_rows)
        return self._record(
            ToolName.INSPECT_SCHEMA,
            {},
            ok=True,
            observation=(
                "Database schema (table names are singular and PascalCase):\n\n"
                f"{schema}"
            ),
        )

    def _execute_sql(self, query: object) -> StepResult:
        if query is None:
            return self._record(
                ToolName.EXECUTE_SQL,
                {"query": None},
                ok=False,
                observation="execute_sql requires a `query` argument.",
            )
        result = self.sandbox.execute(str(query))
        return self._record(
            ToolName.EXECUTE_SQL,
            {"query": str(query)},
            ok=result.ok,
            observation=result.to_observation(),
            query_result=result,
        )

    def _submit_answer(self, answer: object) -> StepResult:
        text = "" if answer is None else str(answer)
        self.trajectory.submitted_answer = text
        result = self._record(
            ToolName.SUBMIT_ANSWER,
            {"answer": text},
            ok=True,
            observation="Answer submitted. Episode complete.",
            terminate=TerminationReason.SUBMITTED,
        )
        return result

    # -- bookkeeping ----------------------------------------------------------

    def _record(
        self,
        tool: ToolName,
        args: dict[str, Any],
        ok: bool,
        observation: str,
        query_result: Any = None,
        terminate: TerminationReason | None = None,
        count_against_budget: bool = True,
    ) -> StepResult:
        record = ToolCallRecord(
            index=self.trajectory.n_calls,
            tool=tool,
            args=args,
            ok=ok,
            observation=observation,
            query_result=query_result,
        )
        if count_against_budget:
            self.trajectory.calls.append(record)

        if terminate is not None:
            self.trajectory.termination = terminate
            return StepResult(observation, True, terminate, record)

        # Budget exhaustion terminates gracefully: the trajectory is complete
        # and gradeable, with no answer and therefore no correctness credit.
        if self.trajectory.n_calls >= self.max_tool_calls:
            self.trajectory.termination = TerminationReason.BUDGET_EXHAUSTED
            return StepResult(
                observation=(
                    f"{observation}\n\n[Tool-call budget of {self.max_tool_calls} "
                    "exhausted. The episode has ended without a submitted answer.]"
                ),
                done=True,
                termination=TerminationReason.BUDGET_EXHAUSTED,
                record=record,
            )

        remaining = self.calls_remaining
        if remaining <= 3:
            observation = (
                f"{observation}\n\n[{remaining} tool call"
                f"{'s' if remaining != 1 else ''} remaining -- submit your answer "
                "before the budget runs out.]"
            )
        return StepResult(observation, False, TerminationReason.RUNNING, record)

    # -- convenience wrappers for scripted agents and adapters ---------------

    def inspect_schema(self) -> str:
        return self.step(ToolName.INSPECT_SCHEMA).observation

    def execute_sql(self, query: str) -> str:
        return self.step(ToolName.EXECUTE_SQL, query=query).observation

    def submit_answer(self, answer: str) -> str:
        return self.step(ToolName.SUBMIT_ANSWER, answer=answer).observation
