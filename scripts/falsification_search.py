#!/usr/bin/env python3
"""Compare uniform vs biased search for finding risky delay settings in Highway-Env."""

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


# DiscreteMetaAction
LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER = 0, 1, 2, 3, 4


@dataclass
class SearchResult:
    method: str
    sample_idx: int
    seed: int
    vehicles_count: int
    delay_steps: int
    delay_ms: float
    collided: int
    failure: int
    min_headway: float
    min_ttc_est: float
    risk_score: float
    steps: int
    best_risk_so_far: float
    failures_found_so_far: int
    first_failure_idx_so_far: int


def heuristic_action(env: gym.Env) -> int:
    vehicle = getattr(env.unwrapped, "vehicle", None)
    road = getattr(env.unwrapped, "road", None)
    if vehicle is None or road is None:
        return IDLE

    target_speed = 22.0
    close_dist = 25.0
    very_close_dist = 15.0

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


def choose_action(policy: str, env: gym.Env) -> int:
    if policy == "heuristic":
        return heuristic_action(env)
    if policy == "idle":
        return IDLE
    if policy == "random":
        return int(env.action_space.sample())
    raise ValueError(f"Unknown policy: {policy}")


def front_vehicle_metrics(env: gym.Env) -> tuple[float, float]:
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
) -> dict:
    _, _ = env.reset(seed=seed)

    action_buffer: Deque[int] = deque([IDLE] * delay_steps, maxlen=delay_steps)

    collided = 0
    steps = 0
    min_headway = float("inf")
    min_ttc_est = float("inf")

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

        headway, ttc_est = front_vehicle_metrics(env)
        min_headway = min(min_headway, headway)
        min_ttc_est = min(min_ttc_est, ttc_est)

        if terminated or truncated:
            break

    near_failure = int(np.isfinite(min_ttc_est) and min_ttc_est < ttc_threshold)
    failure = int(bool(collided) or bool(near_failure))
    risk_score = compute_risk_score(collided, min_headway, min_ttc_est)

    return {
        "collided": collided,
        "failure": failure,
        "min_headway": min_headway if np.isfinite(min_headway) else float("nan"),
        "min_ttc_est": min_ttc_est if np.isfinite(min_ttc_est) else float("nan"),
        "risk_score": risk_score,
        "steps": steps,
        "delay_ms": delay_steps * step_ms,
    }


def sample_setting_uniform(rng: np.random.Generator, delay_values: list[int], traffic_values: list[int]) -> tuple[int, int]:
    delay_steps = int(rng.choice(delay_values))
    vehicles_count = int(rng.choice(traffic_values))
    return delay_steps, vehicles_count


def sample_setting_biased(
    rng: np.random.Generator,
    delay_values: list[int],
    traffic_values: list[int],
) -> tuple[int, int]:
    delay_weights = np.arange(1, len(delay_values) + 1, dtype=float)
    delay_weights /= delay_weights.sum()

    traffic_weights = np.arange(1, len(traffic_values) + 1, dtype=float)
    traffic_weights /= traffic_weights.sum()

    delay_steps = int(rng.choice(delay_values, p=delay_weights))
    vehicles_count = int(rng.choice(traffic_values, p=traffic_weights))
    return delay_steps, vehicles_count


def run_search(
    method: str,
    budget: int,
    delay_values: list[int],
    traffic_values: list[int],
    policy: str,
    max_steps: int,
    policy_frequency: int,
    simulation_frequency: int,
    ttc_threshold: float,
    base_seed: int,
) -> list[SearchResult]:
    rng = np.random.default_rng(base_seed)
    step_ms = 1000.0 / policy_frequency
    results: list[SearchResult] = []

    best_risk_so_far = -float("inf")
    failures_found_so_far = 0
    first_failure_idx_so_far = -1

    env_cache: dict[int, gym.Env] = {}

    try:
        for sample_idx in range(1, budget + 1):
            if method == "uniform":
                delay_steps, vehicles_count = sample_setting_uniform(rng, delay_values, traffic_values)
            elif method == "biased":
                delay_steps, vehicles_count = sample_setting_biased(rng, delay_values, traffic_values)
            else:
                raise ValueError(f"Unknown method: {method}")

            if vehicles_count not in env_cache:
                env = gym.make("highway-v0")
                env.unwrapped.configure(
                    {
                        "vehicles_count": vehicles_count,
                        "policy_frequency": policy_frequency,
                        "simulation_frequency": simulation_frequency,
                        "duration": max(5, int(np.ceil(max_steps / policy_frequency))),
                    }
                )
                env.reset()
                env_cache[vehicles_count] = env

            env = env_cache[vehicles_count]
            seed = int(base_seed * 100000 + sample_idx)

            rollout = run_rollout(
                env=env,
                policy=policy,
                seed=seed,
                delay_steps=delay_steps,
                max_steps=max_steps,
                step_ms=step_ms,
                vehicles_count=vehicles_count,
                ttc_threshold=ttc_threshold,
            )

            best_risk_so_far = max(best_risk_so_far, rollout["risk_score"])
            failures_found_so_far += int(rollout["failure"])
            if first_failure_idx_so_far == -1 and rollout["failure"] == 1:
                first_failure_idx_so_far = sample_idx

            results.append(
                SearchResult(
                    method=method,
                    sample_idx=sample_idx,
                    seed=seed,
                    vehicles_count=vehicles_count,
                    delay_steps=delay_steps,
                    delay_ms=rollout["delay_ms"],
                    collided=rollout["collided"],
                    failure=rollout["failure"],
                    min_headway=rollout["min_headway"],
                    min_ttc_est=rollout["min_ttc_est"],
                    risk_score=rollout["risk_score"],
                    steps=rollout["steps"],
                    best_risk_so_far=best_risk_so_far,
                    failures_found_so_far=failures_found_so_far,
                    first_failure_idx_so_far=first_failure_idx_so_far,
                )
            )
    finally:
        for env in env_cache.values():
            env.close()

    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Uniform vs biased falsification search")
    p.add_argument("--budget", type=int, default=400, help="Number of rollouts per method")
    p.add_argument("--delays", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7])
    p.add_argument("--traffic", type=int, nargs="+", default=[8, 12, 16, 20, 24])
    p.add_argument("--policy", choices=["heuristic", "idle", "random"], default="heuristic")
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--policy-frequency", type=int, default=5)
    p.add_argument("--simulation-frequency", type=int, default=15)
    p.add_argument("--ttc-threshold", type=float, default=1.5)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output", type=Path, default=Path("results/falsification_results.csv"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Running falsification search with budget={args.budget} per method")

    all_results: list[SearchResult] = []
    for method, seed_offset in [("uniform", 0), ("biased", 1000)]:
        print(f"Method: {method}")
        results = run_search(
            method=method,
            budget=args.budget,
            delay_values=args.delays,
            traffic_values=args.traffic,
            policy=args.policy,
            max_steps=args.max_steps,
            policy_frequency=args.policy_frequency,
            simulation_frequency=args.simulation_frequency,
            ttc_threshold=args.ttc_threshold,
            base_seed=args.seed + seed_offset,
        )
        all_results.extend(results)

    df = pd.DataFrame(asdict(r) for r in all_results)
    df.to_csv(args.output, index=False)

    summary = (
        df.groupby("method", as_index=False)
        .agg(
            failures_found=("failure", "sum"),
            collision_count=("collided", "sum"),
            best_risk_found=("risk_score", "max"),
            mean_risk=("risk_score", "mean"),
            first_failure_idx=("first_failure_idx_so_far", "max"),
        )
    )

    top_cases = (
        df.sort_values(["method", "risk_score"], ascending=[True, False])
        .groupby("method")
        .head(5)
        [["method", "sample_idx", "vehicles_count", "delay_steps", "delay_ms", "failure", "collided", "risk_score", "min_ttc_est", "min_headway"]]
    )

    print("Saved search results to", args.output)
    print("\nMethod summary:")
    print(summary.to_string(index=False))

    print("\nTop risky cases per method:")
    print(top_cases.to_string(index=False))


if __name__ == "__main__":
    main()