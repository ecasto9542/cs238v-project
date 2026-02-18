#!/usr/bin/env python3
"""Run actuation-delay robustness experiments in Highway-Env."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Deque

try:
    import gymnasium as gym
    import highway_env  # noqa: F401  # Registers environments.
    import numpy as np
    import pandas as pd
except ModuleNotFoundError as exc:
    missing = exc.name or "a required package"
    raise SystemExit(
        f"Missing dependency: {missing}\n"
        "Install project dependencies first:\n"
        "  python3 -m venv .venv\n"
        "  source .venv/bin/activate\n"
        "  python -m pip install --upgrade pip\n"
        "  pip install -r requirements.txt"
    ) from exc


@dataclass
class RolloutResult:
    seed: int
    vehicles_count: int
    delay_steps: int
    delay_ms: float
    policy: str
    collided: int
    min_ttc: float
    steps: int


def choose_action(policy: str, env: gym.Env) -> int:
    if policy == "random":
        return int(env.action_space.sample())
    if policy == "idle":
        return 1  # DiscreteMetaAction IDLE
    raise ValueError(f"Unknown policy: {policy}")


def get_ttc(info: dict) -> float:
    # If TTC isn't provided by info, return NaN and aggregate with nan-aware min.
    ttc = info.get("time_to_collision")
    if ttc is None:
        return np.nan
    if isinstance(ttc, (list, tuple, np.ndarray)):
        arr = np.asarray(ttc, dtype=float)
        if arr.size == 0:
            return np.nan
        return float(np.nanmin(arr))
    try:
        return float(ttc)
    except (TypeError, ValueError):
        return np.nan


def run_rollout(
    env: gym.Env,
    policy: str,
    seed: int,
    delay_steps: int,
    max_steps: int,
    step_ms: float,
    vehicles_count: int,
) -> RolloutResult:
    _, info = env.reset(seed=seed)

    action_buffer: Deque[int] = deque([1] * delay_steps, maxlen=delay_steps)
    collided = 0
    min_ttc = np.nan

    steps = 0
    for _ in range(max_steps):
        proposed = choose_action(policy, env)
        if delay_steps > 0:
            action_buffer.append(proposed)
            action = action_buffer.popleft()
        else:
            action = proposed

        _, _, terminated, truncated, info = env.step(action)
        steps += 1

        crashed_flag = int(bool(info.get("crashed", False)))
        if hasattr(env.unwrapped, "vehicle"):
            crashed_flag = int(crashed_flag or bool(getattr(env.unwrapped.vehicle, "crashed", False)))
        collided = max(collided, crashed_flag)

        this_ttc = get_ttc(info)
        min_ttc = np.nanmin([min_ttc, this_ttc]) if not np.isnan(this_ttc) else min_ttc

        if terminated or truncated:
            break

    return RolloutResult(
        seed=seed,
        vehicles_count=vehicles_count,
        delay_steps=delay_steps,
        delay_ms=delay_steps * step_ms,
        policy=policy,
        collided=collided,
        min_ttc=float(min_ttc) if not np.isnan(min_ttc) else float("nan"),
        steps=steps,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Actuation delay experiment for highway-env")
    parser.add_argument("--rollouts", type=int, default=50, help="Number of rollouts per setting")
    parser.add_argument("--delays", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="Delay values in environment steps")
    parser.add_argument(
        "--traffic",
        type=int,
        nargs="+",
        default=[20, 35, 50],
        help="Values for vehicles_count in each scenario",
    )
    parser.add_argument("--max-steps", type=int, default=200, help="Maximum steps per rollout")
    parser.add_argument("--policy", choices=["random", "idle"], default="random")
    parser.add_argument("--policy-frequency", type=int, default=5, help="Policy steps per second")
    parser.add_argument("--simulation-frequency", type=int, default=15, help="Physics steps per second")
    parser.add_argument("--output", type=Path, default=Path("results/delay_results.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    all_results: list[RolloutResult] = []

    step_ms = 1000.0 / args.policy_frequency
    total_runs = len(args.traffic) * len(args.delays) * args.rollouts
    completed = 0
    print(f"Starting experiment: {total_runs} rollouts total")

    for vehicles_count in args.traffic:
        env = gym.make("highway-v0")
        env.unwrapped.configure(
            {
                "vehicles_count": vehicles_count,
                "policy_frequency": args.policy_frequency,
                "simulation_frequency": args.simulation_frequency,
                "duration": max(5, int(np.ceil(args.max_steps / args.policy_frequency))),
            }
        )
        env.reset()

        for delay_steps in args.delays:
            print(
                f"Running vehicles_count={vehicles_count}, delay_steps={delay_steps} "
                f"({delay_steps * step_ms:.0f} ms)"
            )
            for rollout_idx in range(args.rollouts):
                seed = 1000 * vehicles_count + 100 * delay_steps + rollout_idx
                result = run_rollout(
                    env,
                    policy=args.policy,
                    seed=seed,
                    delay_steps=delay_steps,
                    max_steps=args.max_steps,
                    step_ms=step_ms,
                    vehicles_count=vehicles_count,
                )
                all_results.append(result)
                completed += 1
                if completed % 25 == 0 or completed == total_runs:
                    print(f"Progress: {completed}/{total_runs} rollouts")

        env.close()

    df = pd.DataFrame(asdict(r) for r in all_results)
    df.to_csv(args.output, index=False)

    summary = (
        df.groupby(["vehicles_count", "delay_steps", "delay_ms"], as_index=False)
        .agg(collision_rate=("collided", "mean"), n=("collided", "count"), mean_steps=("steps", "mean"))
        .sort_values(["vehicles_count", "delay_steps"])
    )

    print("Saved rollout-level results to", args.output)
    print("\nCollision-rate summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
