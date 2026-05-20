#!/usr/bin/env python3
"""
plot_metrics.py

從訓練過程產生的 metrics.csv 繪製各項指標 (對上 epoch)。

用法範例:
  python scripts/plot_metrics.py /path/to/metrics.csv --outdir ./plots --show

支援參數: --metrics 指定逗號分隔的欄位清單，只繪製存在的欄位。
如果未指定，會自動選擇典型的 train 指標。
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd


def infer_metric_columns(df: pd.DataFrame) -> List[str]:
    skip = {"lr-AdamW/pg1", "lr-AdamW/pg2", "step", "epoch"}
    candidates = [c for c in df.columns if c not in skip]
    # prefer commonly used train metrics order
    preferred = [
        "train/loss_ids",
        "train/loss_species",
        "train/acc",
        "train/acc_species",
        "train/mapNone",
        "train/map0.3",
        "train/map0.4",
        "train/map0.5",
        "train/map0.6",
        "train/map0.7",
    ]
    result = [p for p in preferred if p in candidates]
    # append any remaining numeric columns
    for c in candidates:
        if c not in result:
            result.append(c)
    return result


def prepare_dataframe(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # choose rows that contain actual metric logging (have epoch or step)
    if "epoch" in df.columns and df["epoch"].notna().any():
        dfm = df[df["epoch"].notna()].copy()
    elif "step" in df.columns and df["step"].notna().any():
        dfm = df[df["step"].notna()].copy()
    else:
        # fallback: drop rows where all metric columns are NA
        dfm = df.dropna(how="all")

    # coerce numeric columns
    for c in dfm.columns:
        dfm[c] = pd.to_numeric(dfm[c], errors="coerce")

    if "epoch" not in dfm.columns or dfm["epoch"].isna().all():
        # create an epoch column by integerizing the index of metric rows
        dfm = dfm.reset_index(drop=True)
        dfm["epoch"] = (dfm.index).astype(int)
    else:
        dfm["epoch"] = dfm["epoch"].astype(int)

    return dfm


def plot_metrics(
    df: pd.DataFrame, metrics: List[str], outpath: str, show: bool = False
) -> None:
    n = len(metrics)
    if n == 0:
        raise ValueError("沒有可繪製的 metrics")

    cols = 2
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    if isinstance(axes, plt.Axes):
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, m in enumerate(metrics):
        ax = axes[i]
        if m not in df.columns:
            ax.text(0.5, 0.5, f"{m} not found", ha="center", va="center")
            ax.set_title(m)
            ax.axis("off")
            continue
        ax.plot(df["epoch"], df[m], marker="o", linewidth=1)
        ax.set_title(m)
        ax.set_xlabel("epoch")
        ax.set_ylabel(m)
        ax.grid(True, linestyle="--", alpha=0.5)

    # turn off unused axes
    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    fig.savefig(outpath)
    print(f"Saved plots to: {outpath}")
    if show:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot training metrics from metrics.csv"
    )
    parser.add_argument("csv", help="path to metrics.csv")
    parser.add_argument(
        "--metrics", "-m", help="comma-separated metric columns to plot (default: auto)"
    )
    parser.add_argument(
        "--out",
        "-o",
        help="output file path for plot image (default: no save)",
        type=Path,
        default=os.devnull,
    )
    parser.add_argument(
        "--no-show", action="store_true", help="show plots interactively"
    )
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"File not found: {args.csv}")
        sys.exit(2)

    df = prepare_dataframe(args.csv)
    all_metrics = infer_metric_columns(df)

    if args.metrics:
        wanted = [m.strip() for m in args.metrics.split(",") if m.strip() in df.columns]
    else:
        wanted = all_metrics

    if len(wanted) == 0:
        print("沒有發現要繪製的欄位。可用欄位:")
        for c in df.columns:
            print(" -", c)
        sys.exit(1)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    plot_metrics(df, wanted, args.out, show=not args.no_show)


if __name__ == "__main__":
    main()
