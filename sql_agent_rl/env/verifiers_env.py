"""``verifiers`` adapter over the framework-independent episode loop.

Everything substantive -- the sandbox, the tool loop, the rubric -- lives in
plain Python elsewhere in this package. This module only translates: it maps
verifiers' rollout state onto :class:`~sql_agent_rl.env.episode.SQLAgentEpisode`
and exposes each reward component as a separately-weighted, separately-logged
``vf.Rubric`` function.

API notes (verifiers 0.3.x)
---------------------------
The installed package ships two stacks: ``verifiers.v1`` and the classic v0 API,
which also answers at the historical top-level paths (``import verifiers as vf``
gives ``vf.StatefulToolEnv``, ``vf.Rubric``, ...). This adapter targets the
classic API, because it is what the Environments Hub's ``load_environment()``
convention and the TRL/prime-rl integrations expect.

``StatefulToolEnv`` is the right base rather than plain ``ToolEnv``: each
episode needs its own sandbox and task. Its ``add_tool(fn, args_to_skip=[...])``
hides an argument from the schema the model sees, and ``update_tool_args``
injects it at call time -- so the live episode handle is passed to each tool
without ever appearing in the tool definition. Note that the base
``__init__(tools=...)`` path does *not* apply ``args_to_skip``, so the tools are
registered explicitly below instead of being passed to ``super().__init__``.

Import this module only when ``verifiers`` is installed -- the rest of the
package deliberately does not depend on it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import verifiers as vf

from sql_agent_rl.data.chinook import load_chinook
from sql_agent_rl.data.tasks import Difficulty, Task, generate_tasks
from sql_agent_rl.env.episode import (
    DEFAULT_MAX_TOOL_CALLS,
    SYSTEM_PROMPT,
    SQLAgentEpisode,
)
from sql_agent_rl.env.trajectory import ToolName
from sql_agent_rl.rubric.reward import REWARD_COMPONENTS
from sql_agent_rl.sandbox.executor import DEFAULT_MAX_ROWS, DEFAULT_TIMEOUT_S

DEFAULT_DB_PATH = Path("data/chinook.db")

#: Key under which the live episode is stashed in verifiers' rollout state.
EPISODE_KEY = "sql_episode"


class SQLAgentEnv(vf.StatefulToolEnv):
    """Multi-turn SQL analytics environment: one episode per rollout."""

    def __init__(
        self,
        dataset,
        tasks_by_id: dict[str, Task],
        db_path: str | Path = DEFAULT_DB_PATH,
        max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
        max_rows: int = DEFAULT_MAX_ROWS,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        **kwargs: Any,
    ) -> None:
        self.db_path = Path(db_path)
        self.tasks_by_id = tasks_by_id
        self.max_rows = max_rows
        self.timeout_s = timeout_s

        super().__init__(
            tools=[],
            max_turns=max_tool_calls,
            dataset=dataset,
            system_prompt=SYSTEM_PROMPT.format(
                max_rows=max_rows, max_tool_calls=max_tool_calls
            ),
            rubric=build_rubric(),
            **kwargs,
        )
        self.max_tool_calls = max_tool_calls

        # Registered here rather than via `tools=` so that `episode` is stripped
        # from the schema the model sees.
        for tool in (self.inspect_schema, self.execute_sql, self.submit_answer):
            self.add_tool(tool, args_to_skip=["episode"])

    # -- per-rollout episode management --------------------------------------

    async def setup_state(self, state: vf.State) -> vf.State:
        task_id = str(state["info"]["task_id"])
        episode = SQLAgentEpisode(
            task=self.tasks_by_id[task_id],
            db_path=self.db_path,
            max_tool_calls=self.max_tool_calls,
            max_rows=self.max_rows,
            timeout_s=self.timeout_s,
        )
        episode.reset()
        state[EPISODE_KEY] = episode
        return state

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict,
        messages: vf.Messages,
        state: vf.State,
        **kwargs: Any,
    ) -> dict:
        """Inject the live episode, which is hidden from the model's schema."""
        del tool_name, messages, kwargs
        return {**tool_args, "episode": state.get(EPISODE_KEY)}

    @vf.stop
    async def episode_finished(self, state: vf.State) -> bool:
        """Stop as soon as the episode ends (answer submitted, or budget spent).

        Registered as a ``@vf.stop`` predicate rather than by overriding
        ``is_completed``, so it composes with the base class's own stop
        conditions (error, max turns, no tool call) instead of pre-empting them.
        """
        episode = state.get(EPISODE_KEY)
        return episode is not None and episode.done

    # -- tools ----------------------------------------------------------------

    def inspect_schema(self, episode: SQLAgentEpisode | None = None) -> str:
        """Return the database schema: tables, columns, keys and row counts."""
        if episode is None:
            return "Environment error: no active episode."
        return episode.step(ToolName.INSPECT_SCHEMA).observation

    def execute_sql(
        self, query: str, episode: SQLAgentEpisode | None = None
    ) -> str:
        """Run one read-only SQL statement (SELECT/WITH/EXPLAIN) and return rows.

        Args:
            query: A single SQL statement. Semicolon-chained statements are rejected.
        """
        if episode is None:
            return "Environment error: no active episode."
        return episode.step(ToolName.EXECUTE_SQL, query=query).observation

    def submit_answer(
        self, answer: str, episode: SQLAgentEpisode | None = None
    ) -> str:
        """Submit the final answer and end the episode.

        Args:
            answer: The answer, in the format the question asked for.
        """
        if episode is None:
            return "Environment error: no active episode."
        return episode.step(ToolName.SUBMIT_ANSWER, answer=answer).observation


# --------------------------------------------------------------------------
# rubric
# --------------------------------------------------------------------------


def build_rubric() -> vf.Rubric:
    """Expose every reward component as its own named, weighted function.

    verifiers logs each reward function separately, so the per-component
    breakdown appears in eval output and training logs for free. The scalar the
    trainer optimises is the weighted sum, which
    ``tests/test_verifiers_env.py`` asserts is identical to
    :func:`sql_agent_rl.rubric.compute_reward`.

    The grounding gate is expressed here as a ``correctness x grounding``
    product inside the correctness function, because verifiers sums weighted
    reward functions and has no notion of a multiplicative term.
    """
    rubric = vf.Rubric()

    def make(component):
        async def component_fn(state, **kwargs) -> float:
            del kwargs
            episode = state.get(EPISODE_KEY)
            if episode is None:
                return 0.0
            truth = episode.task.ground_truth
            value = float(component.fn(episode.trajectory, truth))
            if component.name == "correctness":
                for gate in REWARD_COMPONENTS:
                    if gate.is_gate:
                        value *= float(gate.fn(episode.trajectory, truth))
            return value

        component_fn.__name__ = component.name
        return component_fn

    for component in REWARD_COMPONENTS:
        fn = make(component)
        if component.is_metric or component.is_gate:
            # Weight 0: logged for inspection, excluded from the reward.
            rubric.add_metric(fn)
        else:
            rubric.add_reward_func(fn, weight=component.weight)
    return rubric


# --------------------------------------------------------------------------
# Environments Hub entry point
# --------------------------------------------------------------------------


def load_environment(
    db_path: str | Path = DEFAULT_DB_PATH,
    num_tasks: int = 200,
    seed: int = 0,
    template_ids: Sequence[str] | None = None,
    difficulties: Sequence[Difficulty | str] | None = None,
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
    max_rows: int = DEFAULT_MAX_ROWS,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    **kwargs: Any,
) -> SQLAgentEnv:
    """Build the environment. This is the entry point verifiers looks for.

    The Chinook database must already be present -- run
    ``python data/download_chinook.py`` once. It is a fixed real dataset, so it
    is fetched as a setup step rather than regenerated per episode.
    """
    from datasets import Dataset

    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(
            f"Chinook database not found at {db_path}. "
            "Run `python data/download_chinook.py` first."
        )

    frames = load_chinook(db_path)
    tasks = generate_tasks(
        frames,
        num_tasks,
        seed=seed,
        template_ids=template_ids,
        difficulties=difficulties,
    )
    tasks_by_id = {f"task-{i:05d}": task for i, task in enumerate(tasks)}

    dataset = Dataset.from_list(
        [
            {
                "question": task.prompt,
                # Never read by the rubric -- ground truth is recomputed in
                # pandas from the task itself. Carried only for eval readability.
                "answer": task.ground_truth.describe(),
                "task": task.template_id,
                "info": {
                    "task_id": task_id,
                    "template_id": task.template_id,
                    "difficulty": str(task.difficulty),
                },
            }
            for task_id, task in tasks_by_id.items()
        ]
    )

    return SQLAgentEnv(
        dataset=dataset,
        tasks_by_id=tasks_by_id,
        db_path=db_path,
        max_tool_calls=max_tool_calls,
        max_rows=max_rows,
        timeout_s=timeout_s,
        **kwargs,
    )
