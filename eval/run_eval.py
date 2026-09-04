#!/usr/bin/env python3
"""Run a real model through N sampled tasks and report per-component rewards.

This is the sanity check to run *before* any training: it exercises the full
loop against a live model and prints the reward breakdown, so you can see
whether the shaping terms behave as intended and whether the grounding gate is
firing on real trajectories rather than only on hand-written ones.

It drives :class:`~sql_agent_rl.env.episode.SQLAgentEpisode` directly over an
OpenAI-compatible chat-completions API, rather than going through verifiers'
rollout machinery. That keeps it dependency-light (any compatible endpoint
works -- vLLM, Together, OpenRouter, OpenAI) and doubles as a demonstration
that the environment is genuinely framework-independent.

Usage:
    export OPENAI_API_KEY=...
    python eval/run_eval.py --model gpt-4o-mini --num-tasks 20

    # against a local vLLM server
    python eval/run_eval.py --model Qwen/Qwen2.5-7B-Instruct \\
        --base-url http://localhost:8000/v1 --num-tasks 50
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sql_agent_rl.data.chinook import load_chinook  # noqa: E402
from sql_agent_rl.data.tasks import Task, generate_tasks  # noqa: E402
from sql_agent_rl.env.episode import (  # noqa: E402
    DEFAULT_MAX_TOOL_CALLS,
    SQLAgentEpisode,
)
from sql_agent_rl.env.trajectory import ToolName  # noqa: E402
from sql_agent_rl.rubric.reward import (  # noqa: E402
    REWARD_COMPONENTS,
    RewardBreakdown,
    compute_reward,
)

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "inspect_schema",
            "description": (
                "Return the database schema: tables, columns, primary and "
                "foreign keys, and row counts."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_sql",
            "description": (
                "Run ONE read-only SQL statement (SELECT/WITH/EXPLAIN) and "
                "return the resulting rows."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A single SQL statement, no semicolon chaining.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": "Submit the final answer and end the episode.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The answer in the format the question requested.",
                    }
                },
                "required": ["answer"],
            },
        },
    },
]


@dataclass
class EpisodeOutcome:
    task: Task
    breakdown: RewardBreakdown
    trajectory_log: dict
    error: str | None = None


async def run_episode(
    client,
    model: str,
    task: Task,
    db_path: Path,
    max_tool_calls: int,
    temperature: float,
) -> EpisodeOutcome:
    """Drive one episode to completion with the model in the loop."""
    with SQLAgentEpisode(
        task, db_path=db_path, max_tool_calls=max_tool_calls
    ) as episode:
        user_prompt = episode.reset()
        messages: list[dict] = [
            {"role": "system", "content": episode.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        error: str | None = None

        # +2 turns of slack so a model that emits a non-tool message still gets
        # nudged back rather than silently stalling the rollout.
        for _ in range(max_tool_calls + 2):
            if episode.done:
                break
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=TOOL_SCHEMAS,
                    temperature=temperature,
                )
            except Exception as exc:  # noqa: BLE001 - report, never abort the sweep
                error = f"{type(exc).__name__}: {exc}"
                break

            message = response.choices[0].message
            calls = message.tool_calls or []
            messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {
                                "name": c.function.name,
                                "arguments": c.function.arguments,
                            },
                        }
                        for c in calls
                    ]
                    or None,
                }
            )
            if not calls:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Please continue by calling one of the available "
                            "tools, or submit_answer if you are ready."
                        ),
                    }
                )
                continue

            for call in calls:
                try:
                    args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                observation = episode.step(call.function.name, **args).observation
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": observation,
                    }
                )
                if episode.done:
                    break

        return EpisodeOutcome(
            task=task,
            breakdown=compute_reward(episode.trajectory),
            trajectory_log=episode.trajectory.to_log_dict(),
            error=error,
        )


def summarise(outcomes: list[EpisodeOutcome]) -> None:
    """Print per-component statistics and per-template correctness."""
    if not outcomes:
        print("No episodes completed.")
        return

    def stats(values: list[float]) -> str:
        mean = statistics.fmean(values)
        stdev = statistics.stdev(values) if len(values) > 1 else 0.0
        return f"{mean:+.3f} +/- {stdev:.3f}"

    totals = [o.breakdown.total for o in outcomes]
    print("\n" + "=" * 72)
    print(f"EPISODES: {len(outcomes)}")
    print("=" * 72)
    print(f"\ntotal reward           {stats(totals)}   "
          f"[min {min(totals):+.3f}, max {max(totals):+.3f}]")

    print("\nper-component (raw, before weighting):")
    for component in REWARD_COMPONENTS:
        values = [o.breakdown.raw[component.name] for o in outcomes]
        print(f"  {component.name:<28} {stats(values)}")

    print("\nper-component (weighted contribution to reward):")
    for component in REWARD_COMPONENTS:
        if component.is_metric or component.is_gate:
            continue
        values = [o.breakdown.weighted[component.name] for o in outcomes]
        print(f"  {component.name:<28} {stats(values)}")

    gated = sum(1 for o in outcomes if o.breakdown.gated)
    solved = sum(1 for o in outcomes if o.breakdown.raw["correctness"] >= 0.999)
    answered = sum(1 for o in outcomes if o.breakdown.raw["answered"] >= 0.5)
    errored = sum(1 for o in outcomes if o.error)
    print(f"\nfully correct          {solved}/{len(outcomes)}")
    print(f"answer submitted       {answered}/{len(outcomes)}")
    print(f"grounding gate fired   {gated}/{len(outcomes)}")
    if errored:
        print(f"API errors             {errored}/{len(outcomes)}")

    print("\nby template:")
    by_template: dict[str, list[EpisodeOutcome]] = {}
    for outcome in outcomes:
        by_template.setdefault(outcome.task.template_id, []).append(outcome)
    for template_id in sorted(by_template):
        group = by_template[template_id]
        correctness = [o.breakdown.raw["correctness"] for o in group]
        reward = [o.breakdown.total for o in group]
        print(
            f"  {template_id:<34} n={len(group):<3} "
            f"correctness={statistics.fmean(correctness):.3f}  "
            f"reward={statistics.fmean(reward):+.3f}"
        )

    print("\nby difficulty:")
    by_difficulty: dict[str, list[EpisodeOutcome]] = {}
    for outcome in outcomes:
        by_difficulty.setdefault(str(outcome.task.difficulty), []).append(outcome)
    for tier in sorted(by_difficulty):
        group = by_difficulty[tier]
        correctness = [o.breakdown.raw["correctness"] for o in group]
        print(
            f"  {tier:<34} n={len(group):<3} "
            f"correctness={statistics.fmean(correctness):.3f}"
        )


async def main_async(args: argparse.Namespace) -> int:
    try:
        from openai import AsyncOpenAI
    except ImportError:
        print(
            "The `openai` package is required for eval. "
            "Install it with: pip install -e '.[eval]'",
            file=sys.stderr,
        )
        return 1

    db_path = Path(args.db_path)
    if not db_path.exists():
        print(
            f"Chinook database not found at {db_path}. "
            "Run `python data/download_chinook.py` first.",
            file=sys.stderr,
        )
        return 1

    api_key = os.environ.get(args.api_key_env)
    if not api_key and not args.base_url:
        print(
            f"No API key found in ${args.api_key_env}. Set it, or pass "
            "--base-url for a local server that does not need one.",
            file=sys.stderr,
        )
        return 1

    frames = load_chinook(db_path)
    tasks = generate_tasks(
        frames,
        args.num_tasks,
        seed=args.seed,
        template_ids=args.templates or None,
        difficulties=args.difficulties or None,
    )

    client = AsyncOpenAI(
        api_key=api_key or "not-needed", base_url=args.base_url or None
    )
    semaphore = asyncio.Semaphore(args.concurrency)

    async def bounded(task: Task) -> EpisodeOutcome:
        async with semaphore:
            return await run_episode(
                client, args.model, task, db_path, args.max_tool_calls,
                args.temperature,
            )

    print(
        f"Running {len(tasks)} episodes against {args.model} "
        f"(concurrency {args.concurrency})..."
    )
    started = time.monotonic()
    outcomes = await asyncio.gather(*(bounded(t) for t in tasks))
    elapsed = time.monotonic() - started

    summarise(list(outcomes))
    print(f"\nelapsed {elapsed:.1f}s")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as handle:
            for outcome in outcomes:
                handle.write(
                    json.dumps(
                        {
                            "model": args.model,
                            **outcome.trajectory_log,
                            **outcome.breakdown.to_log_dict(),
                            "api_error": outcome.error,
                        },
                        default=str,
                    )
                    + "\n"
                )
        print(f"per-episode records written to {out_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-4o-mini", help="Model name.")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", ""),
        help="OpenAI-compatible endpoint (e.g. http://localhost:8000/v1).",
    )
    parser.add_argument(
        "--api-key-env", default="OPENAI_API_KEY",
        help="Environment variable holding the API key.",
    )
    parser.add_argument("--num-tasks", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tool-calls", type=int, default=DEFAULT_MAX_TOOL_CALLS)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--db-path", default="data/chinook.db")
    parser.add_argument(
        "--templates", nargs="*", default=[], help="Restrict to these template ids."
    )
    parser.add_argument(
        "--difficulties", nargs="*", default=[],
        help="Restrict to these tiers: single_table, multi_join, time_comparison.",
    )
    parser.add_argument(
        "--out", default="eval/runs/latest.jsonl",
        help="Where to write per-episode JSONL records ('' to skip).",
    )
    return parser


def main() -> None:
    raise SystemExit(asyncio.run(main_async(build_parser().parse_args())))


if __name__ == "__main__":
    main()
