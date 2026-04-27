"""
Generate PNG figures for 50-case evaluation outputs.

Inputs:
- evaluation_outputs/testset_50/confusion_matrix_50.csv
- evaluation_outputs/testset_50/metrics_summary_50.csv

Outputs:
- evaluation_outputs/testset_50/confusion_matrix_50.png
- evaluation_outputs/testset_50/f1_score_50.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_confusion(path: Path) -> tuple[int, int, int, int]:
    tp = fp = tn = fn = 0
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            actual = (row.get("actual") or "").strip().lower()
            pred_pos = int(float(row.get("predicted_positive") or 0))
            pred_neg = int(float(row.get("predicted_negative") or 0))
            if actual == "positive":
                tp = pred_pos
                fn = pred_neg
            elif actual == "negative":
                fp = pred_pos
                tn = pred_neg
    return tp, fp, tn, fn


def read_metrics(path: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            key = (row.get("metric") or "").strip().lower()
            value = float(row.get("value") or 0.0)
            if key:
                metrics[key] = value
    return metrics


def make_confusion_png(tp: int, fp: int, tn: int, fn: int, out_path: Path) -> None:
    cm = np.array([[tp, fn], [fp, tn]], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks([0, 1], labels=["Predicted Positive", "Predicted Negative"])
    ax.set_yticks([0, 1], labels=["Actual Positive", "Actual Negative"])
    ax.set_title("Confusion Matrix (50-case DDI Test Set)")

    max_v = np.max(cm) if np.max(cm) > 0 else 1.0
    for i in range(2):
        for j in range(2):
            val = int(cm[i, j])
            color = "white" if cm[i, j] > (0.5 * max_v) else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=14, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Count")

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def make_f1_png(metrics: dict[str, float], out_path: Path) -> None:
    f1 = float(metrics.get("f1", 0.0))

    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=150)
    bars = ax.bar(["F1 Score"], [f1], color="#2ca02c", width=0.45)

    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("F1 Score (50-case DDI Test Set)")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f"{h:.4f}", ha="center", va="bottom", fontsize=12)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def make_all_metrics_png(metrics: dict[str, float], out_path: Path) -> None:
    precision = float(metrics.get("precision", 0.0))
    recall = float(metrics.get("recall", 0.0))
    f1 = float(metrics.get("f1", 0.0))
    accuracy = float(metrics.get("accuracy", 0.0))

    labels = ["Precision", "Recall", "F1 Score", "Accuracy"]
    values = [precision, recall, f1, accuracy]
    colors = ["#1f77b4", "#1f77b4", "#1f77b4", "#1f77b4"]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    bars = ax.bar(labels, values, color=colors, width=0.6)

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Performance Metrics (50-case DDI Test Set)", fontsize=14, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    for bar, val in zip(bars, values):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create confusion matrix and F1 PNGs from evaluation CSV outputs")
    parser.add_argument("--input-dir", default="evaluation_outputs/testset_50", help="Folder containing confusion_matrix_50.csv and metrics_summary_50.csv")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    confusion_csv = input_dir / "confusion_matrix_50.csv"
    metrics_csv = input_dir / "metrics_summary_50.csv"

    if not confusion_csv.exists():
        raise FileNotFoundError(f"Missing file: {confusion_csv}")
    if not metrics_csv.exists():
        raise FileNotFoundError(f"Missing file: {metrics_csv}")

    tp, fp, tn, fn = read_confusion(confusion_csv)
    metrics = read_metrics(metrics_csv)

    confusion_png = input_dir / "confusion_matrix_50.png"
    f1_png = input_dir / "f1_score_50.png"
    all_metrics_png = input_dir / "all_metrics_50.png"

    make_confusion_png(tp, fp, tn, fn, confusion_png)
    make_f1_png(metrics, f1_png)
    make_all_metrics_png(metrics, all_metrics_png)

    print(f"Created: {confusion_png}")
    print(f"Created: {f1_png}")
    print(f"Created: {all_metrics_png}")


if __name__ == "__main__":
    main()
