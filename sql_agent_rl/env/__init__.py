"""Multi-turn SQL agent environment (framework-independent core)."""

from sql_agent_rl.env.episode import (
    DEFAULT_MAX_TOOL_CALLS,
    SYSTEM_PROMPT,
    SQLAgentEpisode,
    StepResult,
)
from sql_agent_rl.env.trajectory import (
    TerminationReason,
    ToolCallRecord,
    ToolName,
    Trajectory,
)

__all__ = [
    "DEFAULT_MAX_TOOL_CALLS",
    "SYSTEM_PROMPT",
    "SQLAgentEpisode",
    "StepResult",
    "TerminationReason",
    "ToolCallRecord",
    "ToolName",
    "Trajectory",
]
