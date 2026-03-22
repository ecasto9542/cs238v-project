#!/usr/bin/env python3
"""Estimate failure probability with direct Monte Carlo and importance sampling."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Deque

import gymnasium as gym
import highway_env
import numpy as np
import pandas as pd


LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER = 0, 1, 2, 3, 4


@dataclass
class SampleResult:
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
    weight: float
    cumulative_estimate: float
    cumulative_ci_low: float
    cumulative_ci_high: float


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
        "delay_ms": delay_steps * step_ms,
    }


def make_uniform_probs(values: list[int]) -> dict[int, float]:
    p = 1.0 / len(values)
    return {v: p for v in values}


def make_biased_probs(values: list[int]) -> dict[int, float]:
    weights = np.arange(1, len(values) + 1, dtype=float)
    weights /= weights.sum()
    return {v: float(w) for v, w in zip(values, weights)}


def sample_from_probs(rng: np.random.Generator, probs: dict[int, float]) -> int:
    values = list(probs.keys())
    p = np.array([probs[v] for v in values], dtype=float)
    return int(rng.choice(values, p=p))


def estimate_ci_mc(phat: float, n: int) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    se = np.sqrt(max(phat * (1.0 - phat), 0.0) / n)
    return phat - 1.96 * se, phat + 1.96 * se


def estimate_ci_is(weights: np.ndarray, indicators: np.ndarray) -> tuple[float, float]:
    if len(weights) == 0:
        return float("nan"), float("nan")

    w = np.asarray(weights, dtype=float)
    y = np.asarray(indicators, dtype=float)

    w_sum = w.sum()
    if w_sum <= 0:
        return float("nan"), float("nan")

    w_norm = w / w_sum
    phat = float(np.sum(w_norm * y))

    ess = (w_sum ** 2) / np.sum(w ** 2)
    if ess <= 1:
        return phat, phat

    se = np.sqrt(max(phat * (1.0 - phat), 0.0) / ess)
    return phat - 1.96 * se, phat + 1.96 * se


def run_estimation(
    method: str,
    budget: int,
    delay_probs_nominal: dict[int, float],
    traffic_probs_nominal: dict[int, float],
    delay_probs_proposal: dict[int, float],
    traffic_probs_proposal: dict[int, float],
    policy: str,
    max_steps: int,
    policy_frequency: int,
    simulation_frequency: int,
    ttc_threshold: float,
    base_seed: int,
) -> list[SampleResult]:
    rng = np.random.default_rng(base_seed)
    step_ms = 1000.0 / policy_frequency

    env_cache: dict[int, gym.Env] = {}
    results: list[SampleResult] = []

    failures_mc: list[float] = []
    weights_is: list[float] = []
    indicators_is: list[float] = []

    try:
        for sample_idx in range(1, budget + 1):
            if method == "mc":
                delay_steps = sample_from_probs(rng, delay_probs_nominal)
                vehicles_count = sample_from_probs(rng, traffic_probs_nominal)
                weight = 1.0
            elif method == "is":
                delay_steps = sample_from_probs(rng, delay_probs_proposal)
                vehicles_count = sample_from_probs(rng, traffic_probs_proposal)

                p_nom = delay_probs_nominal[delay_steps] * traffic_probs_nominal[vehicles_count]
                q_prop = delay_probs_proposal[delay_steps] * traffic_probs_proposal[vehicles_count]
                weight = p_nom / q_prop
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

            if method == "mc":
                failures_mc.append(float(rollout["failure"]))
                phat = float(np.mean(failures_mc))
                ci_low, ci_high = estimate_ci_mc(phat, len(failures_mc))
            else:
                weights_is.append(weight)
                indicators_is.append(float(rollout["failure"]))
                w = np.asarray(weights_is, dtype=float)
                y = np.asarray(indicators_is, dtype=float)
                phat = float(np.sum(w * y) / np.sum(w))
                ci_low, ci_high = estimate_ci_is(w, y)

            results.append(
                SampleResult(
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
                    weight=weight,
                    cumulative_estimate=phat,
                    cumulative_ci_low=ci_low,
                    cumulative_ci_high=ci_high,
                )
            )
    finally:
        for env in env_cache.values():
            env.close()

    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Failure probability estimation with MC and IS")
    p.add_argument("--budget", type=int, default=500, help="Number of rollouts per method")
    p.add_argument("--delays", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    p.add_argument("--traffic", type=int, nargs="+", default=[8, 12, 16, 20])
    p.add_argument("--policy", choices=["heuristic", "idle", "random"], default="heuristic")
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--policy-frequency", type=int, default=5)
    p.add_argument("--simulation-frequency", type=int, default=15)
    p.add_argument("--ttc-threshold", type=float, default=1.5)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--output", type=Path, default=Path("results/failure_probability_results.csv"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    delay_values = list(args.delays)
    traffic_values = list(args.traffic)

    delay_probs_nominal = make_uniform_probs(delay_values)
    traffic_probs_nominal = make_uniform_probs(traffic_values)

    delay_probs_proposal = make_biased_probs(delay_values)
    traffic_probs_proposal = make_biased_probs(traffic_values)

    all_results: list[SampleResult] = []

    for method, seed_offset in [("mc", 0), ("is", 1000)]:
        print(f"Running method: {method}")
        results = run_estimation(
            method=method,
            budget=args.budget,
            delay_probs_nominal=delay_probs_nominal,
            traffic_probs_nominal=traffic_probs_nominal,
            delay_probs_proposal=delay_probs_proposal,
            traffic_probs_proposal=traffic_probs_proposal,
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

    summary_rows = []
    for method in ["mc", "is"]:
        sub = df[df["method"] == method].sort_values("sample_idx")
        last = sub.iloc[-1]
        if method == "is":
            w = sub["weight"].to_numpy(dtype=float)
            ess = (w.sum() ** 2) / np.sum(w ** 2)
        else:
            ess = float(len(sub))

        summary_rows.append(
            {
                "method": method,
                "final_estimate": last["cumulative_estimate"],
                "ci_low": last["cumulative_ci_low"],
                "ci_high": last["cumulative_ci_high"],
                "ci_width": last["cumulative_ci_high"] - last["cumulative_ci_low"],
                "ess": ess,
                "failures_observed": int(sub["failure"].sum()),
            }
        )

    summary = pd.DataFrame(summary_rows)
    print("Saved results to", args.output)
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()