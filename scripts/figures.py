#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Plot delay robustness results")
    parser.add_argument("--input", type=Path, default=Path("results/delay_results.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)

    # Aggregate summary (just in case)
    summary = (
        df.groupby(["vehicles_count", "delay_steps", "delay_ms"], as_index=False)
        .agg(
            collision_rate=("collided", "mean"),
            mean_steps=("steps", "mean"),
        )
    )

    sns.set(style="whitegrid")

    # ----------------------------
    # Figure 1: Collision Rate
    # ----------------------------
    plt.figure(figsize=(8,6))
    for traffic in sorted(summary["vehicles_count"].unique()):
        subset = summary[summary["vehicles_count"] == traffic]
        plt.plot(
            subset["delay_ms"],
            subset["collision_rate"],
            marker="o",
            label=f"Traffic = {traffic}"
        )

    plt.xlabel("Delay (ms)")
    plt.ylabel("Collision Rate")
    plt.title("Collision Rate vs Actuation Delay")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_dir / "collision_vs_delay.png", dpi=300)
    plt.close()

    # ----------------------------
    # Figure 2: Mean Steps
    # ----------------------------
    plt.figure(figsize=(8,6))
    for traffic in sorted(summary["vehicles_count"].unique()):
        subset = summary[summary["vehicles_count"] == traffic]
        plt.plot(
            subset["delay_ms"],
            subset["mean_steps"],
            marker="o",
            label=f"Traffic = {traffic}"
        )

    plt.xlabel("Actuation Delay (ms)")
    plt.ylabel("Mean Steps Until Termination")
    plt.title("Mean Episode Length vs Actuation Delay")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_dir / "steps_vs_delay.png", dpi=300)
    plt.close()

    print("Figures saved to", args.output_dir)

if __name__ == "__main__":
    main()