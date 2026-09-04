"""Reward rubric: named components, weights, and the scalar they sum to."""

from sql_agent_rl.rubric.reward import (
    COMPONENTS_BY_NAME,
    MAX_TOTAL_REWARD,
    MIN_TOTAL_REWARD,
    REWARD_COMPONENTS,
    RewardBreakdown,
    RewardComponent,
    compute_reward,
)

__all__ = [
    "COMPONENTS_BY_NAME",
    "MAX_TOTAL_REWARD",
    "MIN_TOTAL_REWARD",
    "REWARD_COMPONENTS",
    "RewardBreakdown",
    "RewardComponent",
    "compute_reward",
]
