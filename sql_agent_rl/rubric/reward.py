"""Weighted assembly of the reward components, with an inspectable breakdown.

Weights and rationale
---------------------

===========================  =======  ===============================================
component                    weight   why
===========================  =======  ===============================================
correctness                   +1.00   the point of the task; dominates everything else
grounding                     gate    multiplies correctness by 0 or 1 (see below)
schema_first_bonus            +0.05   one-time nudge toward looking before leaping
wasted_call_penalty           -0.15   -0.03 per information-free call, capped at five
efficiency_penalty            -0.10   -0.02 per call past eight, capped at five
no_query_attempt_penalty      -0.20   never trying must cost more than trying and failing
===========================  =======  ===============================================

Total lies in ``[-0.45, +1.05]``, and the process terms can add at most ``0.05``
-- 5% of the correctness weight. They are nudges, not the objective.

Grounding is a **multiplier on correctness rather than an additive term**. As an
addend it would be tradeable: an agent could eat the grounding loss and keep the
correctness reward for an answer it never verified. As a gate, an ungrounded
answer is worth exactly zero no matter how right it is, which is the property
the anti-hacking design actually needs.

The rubric returns the full per-component breakdown, not just the scalar, so a
training run can be debugged by looking at *which* term moved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from sql_agent_rl.data.answers import GroundTruth
from sql_agent_rl.env.trajectory import Trajectory
from sql_agent_rl.rubric import components as C

ComponentFn = Callable[[Trajectory, GroundTruth], float]


@dataclass(frozen=True)
class RewardComponent:
    """One named, independently testable term."""

    name: str
    fn: ComponentFn
    weight: float
    #: Metrics are logged but excluded from the total.
    is_metric: bool = False
    #: Gates multiply the correctness term instead of being summed.
    is_gate: bool = False
    description: str = ""


REWARD_COMPONENTS: tuple[RewardComponent, ...] = (
    RewardComponent(
        name="correctness",
        fn=C.correctness,
        weight=1.00,
        description="Match to pandas ground truth computed over the full dataset.",
    ),
    RewardComponent(
        name="grounding",
        fn=C.grounding,
        weight=1.00,
        is_gate=True,
        description="Gate: answer must trace to a query that really read data.",
    ),
    RewardComponent(
        name="schema_first_bonus",
        fn=C.schema_first_bonus,
        weight=0.05,
        description="One-time bonus for inspecting the schema before querying.",
    ),
    RewardComponent(
        name="wasted_call_penalty",
        fn=C.wasted_call_penalty,
        weight=-0.15,
        description=(
            "-0.03 per information-free call (failed query, degenerate query, "
            "or redundant schema inspection), capped at five."
        ),
    ),
    RewardComponent(
        name="efficiency_penalty",
        fn=C.efficiency_penalty,
        weight=-0.10,
        description="-0.02 per tool call past eight, capped at five.",
    ),
    RewardComponent(
        name="no_query_attempt_penalty",
        fn=C.no_query_attempt_penalty,
        weight=-0.20,
        description="Charged when execute_sql was never called at all.",
    ),
    RewardComponent(
        name="answered", fn=C.answered, weight=0.0, is_metric=True,
        description="Whether the episode ended via submit_answer.",
    ),
    RewardComponent(
        name="n_tool_calls", fn=C.n_tool_calls, weight=0.0, is_metric=True,
        description="Tool calls used.",
    ),
    RewardComponent(
        name="n_distinct_tables_read", fn=C.n_distinct_tables_read, weight=0.0,
        is_metric=True, description="Distinct data tables the queries touched.",
    ),
)

COMPONENTS_BY_NAME = {c.name: c for c in REWARD_COMPONENTS}

#: Theoretical bounds, asserted in the tests so a re-weighting cannot silently
#: let process shaping outgrow correctness.
MIN_TOTAL_REWARD = -0.45
MAX_TOTAL_REWARD = 1.05


@dataclass(frozen=True)
class RewardBreakdown:
    """Per-component values plus the final scalar."""

    total: float
    raw: dict[str, float] = field(default_factory=dict)
    weighted: dict[str, float] = field(default_factory=dict)
    gated: bool = False

    def to_log_dict(self) -> dict[str, float | bool]:
        out: dict[str, float | bool] = {"reward": self.total, "gated": self.gated}
        out.update({f"raw/{k}": v for k, v in self.raw.items()})
        out.update({f"weighted/{k}": v for k, v in self.weighted.items()})
        return out

    def __str__(self) -> str:
        parts = ", ".join(f"{k}={v:+.3f}" for k, v in self.weighted.items() if v)
        gate = " [GATED]" if self.gated else ""
        return f"reward={self.total:+.3f}{gate} ({parts or 'no non-zero terms'})"


def compute_reward(
    trajectory: Trajectory, truth: GroundTruth | None = None
) -> RewardBreakdown:
    """Score one episode, returning every component alongside the total."""
    if truth is None:
        truth = trajectory.task.ground_truth

    raw = {c.name: float(c.fn(trajectory, truth)) for c in REWARD_COMPONENTS}

    gate = 1.0
    for component in REWARD_COMPONENTS:
        if component.is_gate:
            gate *= raw[component.name]

    weighted: dict[str, float] = {}
    total = 0.0
    for component in REWARD_COMPONENTS:
        if component.is_metric or component.is_gate:
            weighted[component.name] = 0.0
            continue
        value = raw[component.name] * component.weight
        # The gate suppresses the correctness term only; process shaping is
        # charged either way, so an ungrounded episode still pays for its
        # wasted calls rather than being silently zeroed out.
        if component.name == "correctness":
            value *= gate
        weighted[component.name] = value
        total += value

    return RewardBreakdown(
        total=total, raw=raw, weighted=weighted, gated=gate == 0.0
    )
