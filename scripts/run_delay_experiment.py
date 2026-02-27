#!/usr/bin/env python3
"""Run actuation-delay robustness experiments in Highway-Env (non-random baseline)."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Deque

import gymnasium as gym
import highway_env  # registers envs
import numpy as np
import pandas as pd


# Highway-Env DiscreteMetaAction mapping is typically:
# 0: LANE_LEFT, 1: IDLE, 2: LANE_RIGHT, 3: FASTER, 4: SLOWER
LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER = 0, 1, 2, 3, 4


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


def get_min_ttc_from_info(info: dict) -> float:
    """Best-effort TTC extraction; returns NaN if env doesn't provide it."""
    ttc = info.get("time_to_collision")
    if ttc is None:
        return float("nan")
    if isinstance(ttc, (list, tuple, np.ndarray)):
        arr = np.asarray(ttc, dtype=float)
        return float(np.nanmin(arr)) if arr.size else float("nan")
    try:
        return float(ttc)
    except (TypeError, ValueError):
        return float("nan")


def choose_action(policy: str, env: gym.Env) -> int:
    if policy == "random":
        return int(env.action_space.sample())
    if policy == "idle":
        return IDLE
    if policy == "heuristic":
        return heuristic_action(env)
    raise ValueError(f"Unknown policy: {policy}")


def heuristic_action(env: gym.Env) -> int:
    vehicle = getattr(env.unwrapped, "vehicle", None)
    road = getattr(env.unwrapped, "road", None)
    if vehicle is None or road is None:
        return IDLE

    # Conservative settings
    target_speed = 22.0       # m/s (~49 mph)
    close_dist = 25.0         # start slowing
    very_close_dist = 15.0    # definitely slow

    my_lane = vehicle.lane_index
    my_x = vehicle.position[0]

    front_dist = float("inf")
    for other in road.vehicles:
        if other is vehicle:
            continue
        if getattr(other, "lane_index", None) != my_lane:
            continue
        dx = other.position[0] - my_x
        if 0 < dx < front_dist:
            front_dist = dx

    # If close to front car: slow down
    if front_dist < very_close_dist:
        return SLOWER
    if front_dist < close_dist:
        return SLOWER

    # Otherwise speed control
    if getattr(vehicle, "speed", target_speed) < target_speed - 1.5:
        return FASTER
    if getattr(vehicle, "speed", target_speed) > target_speed + 2.5:
        return SLOWER
    return IDLE


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

    # Delay buffer: prefill with IDLE so the car "does nothing" while actions are delayed in
    action_buffer: Deque[int] = deque([IDLE] * delay_steps, maxlen=delay_steps)

    collided = 0
    min_ttc = float("nan")
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

        this_ttc = get_min_ttc_from_info(info)
        if not np.isnan(this_ttc):
            min_ttc = this_ttc if np.isnan(min_ttc) else float(min(min_ttc, this_ttc))

        if terminated or truncated:
            break

    return RolloutResult(
        seed=seed,
        vehicles_count=vehicles_count,
        delay_steps=delay_steps,
        delay_ms=delay_steps * step_ms,
        policy=policy,
        collided=collided,
        min_ttc=min_ttc,
        steps=steps,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Actuation delay experiments for highway-env")
    p.add_argument("--rollouts", type=int, default=100)
    p.add_argument("--delays", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--traffic", type=int, nargs="+", default=[15, 25, 35])
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--policy", choices=["heuristic", "idle", "random"], default="heuristic")
    p.add_argument("--policy-frequency", type=int, default=5)
    p.add_argument("--simulation-frequency", type=int, default=15)
    p.add_argument("--output", type=Path, default=Path("results/delay_results.csv"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    step_ms = 1000.0 / args.policy_frequency
    total_runs = len(args.traffic) * len(args.delays) * args.rollouts
    print(f"Starting experiment: {total_runs} rollouts total")

    all_results: list[RolloutResult] = []
    completed = 0

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
            print(f"Running vehicles_count={vehicles_count}, delay_steps={delay_steps} ({delay_steps * step_ms:.0f} ms)")
            for rollout_idx in range(args.rollouts):
                seed = 1000 * vehicles_count + 100 * delay_steps + rollout_idx
                all_results.append(
                    run_rollout(
                        env=env,
                        policy=args.policy,
                        seed=seed,
                        delay_steps=delay_steps,
                        max_steps=args.max_steps,
                        step_ms=step_ms,
                        vehicles_count=vehicles_count,
                    )
                )
                completed += 1
                if completed % 50 == 0 or completed == total_runs:
                    print(f"Progress: {completed}/{total_runs}")

        env.close()

    df = pd.DataFrame(asdict(r) for r in all_results)
    df.to_csv(args.output, index=False)

    summary = (
        df.groupby(["vehicles_count", "delay_steps", "delay_ms"], as_index=False)
        .agg(
            collision_rate=("collided", "mean"),
            n=("collided", "count"),
            mean_steps=("steps", "mean"),
            mean_min_ttc=("min_ttc", "mean"),
        )
        .sort_values(["vehicles_count", "delay_steps"])
    )

    print("Saved rollout-level results to", args.output)
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()