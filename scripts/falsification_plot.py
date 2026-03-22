#!/usr/bin/env python3
"""Plot falsification search results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot falsification results")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/falsification_results.csv"),
        help="CSV file produced by falsification_search.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/figures"),
        help="Directory to save plots",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    df = df.sort_values(["method", "sample_idx"])

    # Figure 1: cumulative failures found
    plt.figure(figsize=(8, 5.5))
    for method in sorted(df["method"].unique()):
        subset = df[df["method"] == method]
        plt.plot(
            subset["sample_idx"],
            subset["failures_found_so_far"],
            marker="o",
            markersize=3,
            linewidth=2,
            label=method.capitalize(),
        )
    plt.xlabel("Search Budget (Number of Rollouts)")
    plt.ylabel("Cumulative Failures Found")
    plt.title("Falsification Efficiency: Failures Found vs Search Budget")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_dir / "falsification_failures_vs_budget.png", dpi=300)
    plt.close()

    # Figure 2: cumulative best non-saturated risk proxy
    # Since risk_score saturates at 1e6 on collision, use cumulative min TTC among failures/nonfailures
    # Smaller TTC is riskier, so we track cumulative minimum TTC seen so far.
    plt.figure(figsize=(8, 5.5))
    for method in sorted(df["method"].unique()):
        subset = df[df["method"] == method].copy()
        subset["ttc_for_plot"] = subset["min_ttc_est"].fillna(float("inf"))
        subset["best_min_ttc_so_far"] = subset["ttc_for_plot"].cummin()
        plt.plot(
            subset["sample_idx"],
            subset["best_min_ttc_so_far"],
            marker="o",
            markersize=3,
            linewidth=2,
            label=method.capitalize(),
        )
    plt.xlabel("Search Budget (Number of Rollouts)")
    plt.ylabel("Best (Lowest) TTC Found So Far")
    plt.title("Falsification Efficiency: Lowest TTC Found vs Search Budget")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_dir / "falsification_ttc_vs_budget.png", dpi=300)
    plt.close()

    # Optional quick summary printed to terminal
    summary = (
        df.groupby("method", as_index=False)
        .agg(
            failures_found=("failure", "sum"),
            first_failure_idx=("first_failure_idx_so_far", "max"),
            min_ttc_found=("min_ttc_est", "min"),
            max_delay_found=("delay_steps", "max"),
            max_traffic_found=("vehicles_count", "max"),
        )
    )
    print("Saved figures to", args.output_dir)
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()