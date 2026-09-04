"""Episode trajectory: the complete record the reward rubric grades.

Every reward component is a pure function of ``(Trajectory, GroundTruth)``.
Keeping the trajectory a plain, serialisable dataclass -- with no reference to
a live database, model or RL framework -- is what makes that possible: the
adversarial tests in ``tests/test_rubric.py`` construct trajectories by hand
and score them without an LLM anywhere in the loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from sql_agent_rl.data.tasks import Task
from sql_agent_rl.sandbox.executor import QueryResult


class ToolName(str, Enum):
    INSPECT_SCHEMA = "inspect_schema"
    EXECUTE_SQL = "execute_sql"
    SUBMIT_ANSWER = "submit_answer"
    #: A call to a tool that does not exist. Recorded under its own name so a
    #: bogus tool name burns budget without being miscounted as a real call --
    #: filing it under INSPECT_SCHEMA would hand out the schema-first bonus.
    UNKNOWN = "unknown"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


class TerminationReason(str, Enum):
    #: The agent called ``submit_answer``.
    SUBMITTED = "submitted"
    #: The tool-call budget ran out first. Terminates gracefully with no
    #: answer, rather than raising.
    BUDGET_EXHAUSTED = "budget_exhausted"
    #: Still running.
    RUNNING = "running"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


@dataclass(frozen=True)
class ToolCallRecord:
    """One tool call and what the agent saw back."""

    index: int
    tool: ToolName
    args: dict[str, Any]
    ok: bool
    observation: str
    #: Only populated for ``execute_sql``.
    query_result: QueryResult | None = None

    @property
    def is_query(self) -> bool:
        return self.tool is ToolName.EXECUTE_SQL

    @property
    def query_succeeded(self) -> bool:
        return self.query_result is not None and self.query_result.ok

    @property
    def touched_real_tables(self) -> bool:
        """Whether SQLite actually resolved a base table for this query.

        Reported by the sandbox's authorizer, not inferred from the SQL text.
        """
        return bool(self.query_result and self.query_result.touched_real_tables)


@dataclass
class Trajectory:
    """Ordered history of one episode."""

    task: Task
    max_tool_calls: int
    calls: list[ToolCallRecord] = field(default_factory=list)
    submitted_answer: str | None = None
    termination: TerminationReason = TerminationReason.RUNNING

    # -- counts ---------------------------------------------------------------

    @property
    def n_calls(self) -> int:
        return len(self.calls)

    @property
    def schema_inspections(self) -> list[ToolCallRecord]:
        return [c for c in self.calls if c.tool is ToolName.INSPECT_SCHEMA]

    @property
    def query_attempts(self) -> list[ToolCallRecord]:
        """Every ``execute_sql`` call, successful or not."""
        return [c for c in self.calls if c.is_query]

    @property
    def successful_queries(self) -> list[ToolCallRecord]:
        return [c for c in self.query_attempts if c.query_succeeded]

    @property
    def failed_queries(self) -> list[ToolCallRecord]:
        return [c for c in self.query_attempts if not c.query_succeeded]

    @property
    def grounded_queries(self) -> list[ToolCallRecord]:
        """Successful queries that actually read at least one data table.

        ``SELECT 'Acme Corp'`` and ``SELECT * FROM sqlite_master`` are excluded:
        they succeed, but they read no data.
        """
        return [c for c in self.successful_queries if c.touched_real_tables]

    @property
    def degenerate_queries(self) -> list[ToolCallRecord]:
        """Queries that succeeded without reading any data table.

        Counted alongside errors by the error-penalty component: a call that
        reads nothing is a wasted call, and treating it as free would leave an
        opening to satisfy "you executed a query" with ``SELECT 1``.
        """
        return [c for c in self.successful_queries if not c.touched_real_tables]

    @property
    def redundant_schema_inspections(self) -> list[ToolCallRecord]:
        """Schema inspections after the first.

        The schema does not change during an episode, so every repeat returns
        byte-identical text: an information-free call, exactly like a
        successful ``SELECT 1``. Counted as wasted so that farming the
        schema bonus is strictly loss-making rather than merely capped.
        """
        return self.schema_inspections[1:]

    @property
    def inspected_schema_before_first_query(self) -> bool:
        first_inspect = next(
            (c.index for c in self.calls if c.tool is ToolName.INSPECT_SCHEMA), None
        )
        if first_inspect is None:
            return False
        first_query = next((c.index for c in self.calls if c.is_query), None)
        return first_query is None or first_inspect < first_query

    @property
    def tables_read(self) -> set[str]:
        """Union of every data table read across the episode."""
        return {
            table
            for call in self.grounded_queries
            if call.query_result
            for table in call.query_result.tables_read
        }

    def result_rows(self) -> list[tuple[Any, ...]]:
        """Every row the agent actually saw from a grounded successful query."""
        return [
            row
            for call in self.grounded_queries
            if call.query_result
            for row in call.query_result.rows
        ]

    def to_log_dict(self) -> dict[str, Any]:
        """Compact, JSON-serialisable summary for eval logs."""
        return {
            "template_id": self.task.template_id,
            "difficulty": str(self.task.difficulty),
            "params": self.task.params,
            "termination": str(self.termination),
            "n_calls": self.n_calls,
            "n_schema_inspections": len(self.schema_inspections),
            "n_query_attempts": len(self.query_attempts),
            "n_successful_queries": len(self.successful_queries),
            "n_grounded_queries": len(self.grounded_queries),
            "n_failed_queries": len(self.failed_queries),
            "n_degenerate_queries": len(self.degenerate_queries),
            "tables_read": sorted(self.tables_read),
            "submitted_answer": self.submitted_answer,
            "ground_truth": self.task.ground_truth.describe(),
            "calls": [
                {
                    "i": c.index,
                    "tool": str(c.tool),
                    "ok": c.ok,
                    "sql": c.args.get("query") if c.is_query else None,
                    "error_type": (
                        str(c.query_result.error_type)
                        if c.query_result and c.query_result.error_type
                        else None
                    ),
                    "rows": c.query_result.row_count if c.query_result else None,
                }
                for c in self.calls
            ],
        }
