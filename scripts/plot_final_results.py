#!/usr/bin/env python3
"""Generate final plots for delay robustness, falsification, and failure estimation."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_direct_sampling(delay_csv: Path, output_dir: Path) -> None:
    df = pd.read_csv(delay_csv)

    summary = (
        df.groupby(["vehicles_count", "delay_steps", "delay_ms"], as_index=False)
        .agg(
            failure_rate=("failure", "mean") if "failure" in df.columns else ("collided", "mean"),
            collision_rate=("collided", "mean"),
            n=("collided", "count"),
        )
        .sort_values(["vehicles_count", "delay_steps"])
    )

    plt.figure(figsize=(8, 5.5))
    for traffic in sorted(summary["vehicles_count"].unique()):
        subset = summary[summary["vehicles_count"] == traffic]
        plt.plot(
            subset["delay_ms"],
            subset["failure_rate"],
            marker="o",
            linewidth=2,
            label=f"Traffic = {traffic}",
        )

    plt.xlabel("Control Delay (ms)")
    plt.ylabel("Failure Rate")
    plt.title("Direct Sampling: Failure Rate vs Control Delay")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "direct_sampling_failure_vs_delay.png", dpi=300)
    plt.close()


def plot_falsification(falsification_csv: Path, output_dir: Path) -> None:
    df = pd.read_csv(falsification_csv).sort_values(["method", "sample_idx"])

    plt.figure(figsize=(8, 5.5))
    for method in sorted(df["method"].unique()):
        subset = df[df["method"] == method]
        plt.plot(
            subset["sample_idx"],
            subset["failures_found_so_far"],
            marker="o",
            markersize=3,
            linewidth=2,
            label = "Nominal Sampling" if method == "uniform" else "Fuzzing"
        )

    plt.xlabel("Search Budget (Number of Rollouts)")
    plt.ylabel("Cumulative Failures Found")
    plt.title("Falsification Efficiency: Failures Found vs Search Budget")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "falsification_failures_vs_budget.png", dpi=300)
    plt.close()


def plot_failure_estimation(estimation_csv: Path, output_dir: Path) -> None:
    df = pd.read_csv(estimation_csv).sort_values(["method", "sample_idx"])

    plt.figure(figsize=(8, 5.5))
    

    for method in sorted(df["method"].unique()):
        subset = df[df["method"] == method]
        lower = np.maximum(subset["cumulative_ci_low"], 0)
        upper = np.minimum(subset["cumulative_ci_high"], 1)
        plt.plot(
            subset["sample_idx"],
            subset["cumulative_estimate"],
            linewidth=2,
            label=method.upper(),
        )
        plt.fill_between(
            subset["sample_idx"],
            lower,
            upper,
            alpha=0.2,
        )

    plt.xlabel("Number of Rollouts")
    plt.ylabel("Estimated Failure Probability")
    plt.title("Failure Probability Estimation: MC vs IS")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "failure_probability_estimate_vs_budget.png", dpi=300)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate final project plots")
    parser.add_argument(
        "--delay-results",
        type=Path,
        default=Path("results/delay_results_phase1.csv"),
        help="CSV from run_delay_experiment.py",
    )
    parser.add_argument(
        "--falsification-results",
        type=Path,
        default=Path("results/falsification_results.csv"),
        help="CSV from falsification_search.py",
    )
    parser.add_argument(
        "--failure-estimation-results",
        type=Path,
        default=Path("results/failure_probability_results.csv"),
        help="CSV from failure_probability_estimation.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/final_figures"),
        help="Directory to save all plots",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_direct_sampling(args.delay_results, args.output_dir)
    plot_falsification(args.falsification_results, args.output_dir)
    plot_failure_estimation(args.failure_estimation_results, args.output_dir)

    print("Saved final plots to", args.output_dir)


if __name__ == "__main__":
    main()
