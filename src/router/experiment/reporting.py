"""Result/figure/table generation from analysis outputs.

Table generation (CSV/markdown) has no extra dependency. Figure generation
needs `matplotlib` (the `viz` extra — `pip install -e ".[viz]"`) and is
imported lazily inside each plotting function so the rest of the package
works without it installed.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from router.analysis.frontier import FrontierPoint
from router.analysis.thresholds import ThresholdResult


def rows_from_models(models: Sequence[BaseModel]) -> list[dict[str, Any]]:
    return [m.model_dump() for m in models]


def write_csv_table(rows: list[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def rows_to_markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "(no rows)"
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)


def threshold_results_table(results: list[ThresholdResult], path: Path) -> Path:
    return write_csv_table(rows_from_models(results), path)


def _matplotlib_pyplot():
    import matplotlib

    matplotlib.use("Agg")  # headless — this module never opens a window
    import matplotlib.pyplot as plt

    return plt


def plot_cost_quality_frontier(
    all_points: list[FrontierPoint],
    pareto_points: list[FrontierPoint],
    output_path: Path,
    title: str = "Cost-quality frontier",
) -> Path:
    plt = _matplotlib_pyplot()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(
        [p.avg_cost for p in all_points], [p.avg_quality for p in all_points], alpha=0.4, label="all tau"
    )
    pareto_sorted = sorted(pareto_points, key=lambda p: p.avg_cost)
    ax.plot(
        [p.avg_cost for p in pareto_sorted],
        [p.avg_quality for p in pareto_sorted],
        marker="o",
        label="Pareto frontier",
    )
    ax.set_xlabel("Average cost per request")
    ax.set_ylabel("Average quality (q_hat)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_reliability_diagram(
    y_true: list[int],
    probs: list[float],
    output_path: Path,
    n_bins: int = 10,
    title: str = "Calibration reliability diagram",
) -> Path:
    import numpy as np

    plt = _matplotlib_pyplot()

    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probs, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    bin_centers, bin_accuracies = [], []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (p >= lo) & (p <= hi) if i == n_bins - 1 else (p >= lo) & (p < hi)
        if not mask.any():
            continue
        bin_centers.append(float(p[mask].mean()))
        bin_accuracies.append(float(y[mask].mean()))

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    ax.plot(bin_centers, bin_accuracies, marker="o", label="observed")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
