#!/usr/bin/env python3
"""Run actuation-delay validation experiments in Highway-Env."""

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
ACTION_NAMES = {
    LANE_LEFT: "LANE_LEFT",
    IDLE: "IDLE",
    LANE_RIGHT: "LANE_RIGHT",
    FASTER: "FASTER",
    SLOWER: "SLOWER",
}


@dataclass
class RolloutResult:
    seed: int
    vehicles_count: int
    delay_steps: int
    delay_ms: float
    policy: str
    collided: int
    failure: int
    min_headway: float
    min_ttc_est: float
    risk_score: float
    steps: int


def choose_action(policy: str, env: gym.Env) -> int:
    if policy == "random":
        return int(env.action_space.sample())
    if policy == "idle":
        return IDLE
    if policy == "heuristic":
        return heuristic_action(env)
    raise ValueError(f"Unknown policy: {policy}")


def heuristic_action(env: gym.Env) -> int:
    """Simple deterministic baseline controller."""
    vehicle = getattr(env.unwrapped, "vehicle", None)
    road = getattr(env.unwrapped, "road", None)
    if vehicle is None or road is None:
        return IDLE

    target_speed = 22.0       # m/s
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

    if front_dist < very_close_dist:
        return SLOWER
    if front_dist < close_dist:
        return SLOWER

    if getattr(vehicle, "speed", target_speed) < target_speed - 1.5:
        return FASTER
    if getattr(vehicle, "speed", target_speed) > target_speed + 2.5:
        return SLOWER
    return IDLE


def front_vehicle_metrics(env: gym.Env) -> tuple[float, float]:
    """
    Returns:
        headway: distance to closest front vehicle in same lane
        ttc_est: estimated TTC = distance / closing_speed if closing_speed > 0, else inf
    """
    vehicle = getattr(env.unwrapped, "vehicle", None)
    road = getattr(env.unwrapped, "road", None)
    if vehicle is None or road is None:
        return float("inf"), float("inf")

    my_lane = vehicle.lane_index
    my_x = float(vehicle.position[0])
    my_speed = float(getattr(vehicle, "speed", 0.0))

    front = None
    front_dist = float("inf")

    for other in road.vehicles:
        if other is vehicle:
            continue
        if getattr(other, "lane_index", None) != my_lane:
            continue
        dx = float(other.position[0]) - my_x
        if 0 < dx < front_dist:
            front_dist = dx
            front = other

    if front is None:
        return float("inf"), float("inf")

    front_speed = float(getattr(front, "speed", 0.0))
    closing_speed = my_speed - front_speed

    if closing_speed > 1e-6:
        ttc_est = front_dist / closing_speed
    else:
        ttc_est = float("inf")

    return front_dist, ttc_est


def compute_risk_score(collided: int, min_headway: float, min_ttc_est: float) -> float:
    """
    Continuous surrogate objective for falsification.
    Higher means more dangerous.
    """
    if collided:
        return 1e6

    headway_term = 0.0 if not np.isfinite(min_headway) else 1.0 / max(min_headway, 1e-3)
    ttc_term = 0.0 if not np.isfinite(min_ttc_est) else 1.0 / max(min_ttc_est, 1e-3)

    return max(headway_term, 2.0 * ttc_term)


def run_rollout(
    env: gym.Env,
    policy: str,
    seed: int,
    delay_steps: int,
    max_steps: int,
    step_ms: float,
    vehicles_count: int,
    ttc_threshold: float,
    inspect_timestep: int | None = None,
) -> RolloutResult:
    _, info = env.reset(seed=seed)

    action_buffer: Deque[int] = deque([IDLE] * delay_steps, maxlen=delay_steps)

    collided = 0
    steps = 0
    min_headway = float("inf")
    min_ttc_est = float("inf")

    for _ in range(max_steps):
        proposed = choose_action(policy, env)

        if delay_steps > 0:
            buffer_before_append = list(action_buffer)
            action_buffer.append(proposed)
            buffer_after_append = list(action_buffer)
            action = action_buffer.popleft()
            buffer_after_pop = list(action_buffer)
        else:
            buffer_before_append = []
            buffer_after_append = []
            buffer_after_pop = []
            action = proposed

        if inspect_timestep is not None and steps == inspect_timestep:
            print("\n--- Delay Buffer Inspection ---")
            print(f"seed={seed}, vehicles_count={vehicles_count}, delay_steps={delay_steps}")
            print(f"timestep={steps}")
            print(f"proposed action={ACTION_NAMES.get(proposed, proposed)}")
            print(f"buffer before append={[ACTION_NAMES.get(a, a) for a in buffer_before_append]}")
            print(f"buffer after append={[ACTION_NAMES.get(a, a) for a in buffer_after_append]}")
            print(f"executed action={ACTION_NAMES.get(action, action)}")
            print(f"buffer after pop={[ACTION_NAMES.get(a, a) for a in buffer_after_pop]}")
            print("-------------------------------\n")

        _, _, terminated, truncated, info = env.step(action)
        steps += 1

        crashed_flag = int(bool(info.get("crashed", False)))
        if hasattr(env.unwrapped, "vehicle"):
            crashed_flag = int(crashed_flag or bool(getattr(env.unwrapped.vehicle, "crashed", False)))
        collided = max(collided, crashed_flag)

        headway, ttc_est = front_vehicle_metrics(env)
        min_headway = min(min_headway, headway)
        min_ttc_est = min(min_ttc_est, ttc_est)

        if terminated or truncated:
            break

    near_failure = int(np.isfinite(min_ttc_est) and min_ttc_est < ttc_threshold)
    failure = int(bool(collided) or bool(near_failure))

    risk_score = compute_risk_score(
        collided=collided,
        min_headway=min_headway,
        min_ttc_est=min_ttc_est,
    )

    return RolloutResult(
        seed=seed,
        vehicles_count=vehicles_count,
        delay_steps=delay_steps,
        delay_ms=delay_steps * step_ms,
        policy=policy,
        collided=collided,
        failure=failure,
        min_headway=min_headway if np.isfinite(min_headway) else float("nan"),
        min_ttc_est=min_ttc_est if np.isfinite(min_ttc_est) else float("nan"),
        risk_score=risk_score,
        steps=steps,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Actuation delay experiments for highway-env")
    p.add_argument("--rollouts", type=int, default=100)
    p.add_argument("--delays", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    p.add_argument("--traffic", type=int, nargs="+", default=[8, 12, 16, 20])
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--policy", choices=["heuristic", "idle", "random"], default="heuristic")
    p.add_argument("--policy-frequency", type=int, default=5)
    p.add_argument("--simulation-frequency", type=int, default=15)
    p.add_argument("--ttc-threshold", type=float, default=1.5, help="Failure if estimated TTC drops below this")
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
            print(
                f"Running vehicles_count={vehicles_count}, "
                f"delay_steps={delay_steps} ({delay_steps * step_ms:.0f} ms)"
            )
            for rollout_idx in range(args.rollouts):
                seed = 1000 * vehicles_count + 100 * delay_steps + rollout_idx
                result = run_rollout(
                    env=env,
                    policy=args.policy,
                    seed=seed,
                    delay_steps=delay_steps,
                    max_steps=args.max_steps,
                    step_ms=step_ms,
                    vehicles_count=vehicles_count,
                    ttc_threshold=args.ttc_threshold,
                    inspect_timestep=args.inspect_timestep,
                )
                all_results.append(result)
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
            failure_rate=("failure", "mean"),
            mean_steps=("steps", "mean"),
            mean_min_headway=("min_headway", "mean"),
            mean_min_ttc_est=("min_ttc_est", "mean"),
            mean_risk_score=("risk_score", "mean"),
            max_risk_score=("risk_score", "max"),
            n=("failure", "count"),
        )
        .sort_values(["vehicles_count", "delay_steps"])
    )

    print("Saved rollout-level results to", args.output)
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()