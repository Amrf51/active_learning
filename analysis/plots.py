"""
plots.py — thesis-ready figures from grouped AL runs.

All figures plot against the LABEL BUDGET (labeled_pool_size) on the x-axis so runs with
different cycle counts stay comparable, and show the mean line + std band across seeds.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless — no display needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .aggregate import RunRecord, dedupe_by_seed  # noqa: E402

logger = logging.getLogger("analysis.plots")

# stable color per strategy across all figures
_STRATEGY_COLORS = {
    "random": "#7f7f7f",
    "entropy": "#1f77b4",
    "margin": "#2ca02c",
    "least_confidence": "#d62728",
    "uncertainty": "#d62728",
}


def _color(strategy: str) -> str:
    return _STRATEGY_COLORS.get(strategy, None)


def _mean_band(runs: list[RunRecord], metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (grid, mean, std) for `metric` across seeds, interpolated to a common grid.

    The grid is the run with the most points (densest budget sampling); each seed's curve
    is linearly interpolated onto it before averaging. With a single seed, std is 0.
    """
    series = []
    for r in runs:
        x = r.labels
        y = getattr(r, metric)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() >= 1:
            series.append((x[m], y[m]))
    if not series:
        return np.array([]), np.array([]), np.array([])

    grid = max((x for x, _ in series), key=len)
    grid = np.sort(np.unique(grid))
    stacked = []
    for x, y in series:
        order = np.argsort(x)
        stacked.append(np.interp(grid, x[order], y[order]))
    arr = np.vstack(stacked)
    return grid, arr.mean(axis=0), arr.std(axis=0)


def _curve_figure(
    groups_for_dataset: dict[str, list[RunRecord]],
    metric: str,
    ylabel: str,
    title: str,
    out_path: Path,
    baseline: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for strategy in sorted(groups_for_dataset):
        runs = dedupe_by_seed(groups_for_dataset[strategy])
        grid, mean, std = _mean_band(runs, metric)
        if grid.size == 0:
            continue
        n = len(runs)
        c = _color(strategy)
        label = f"{strategy} (n={n})"
        ax.plot(grid, mean, label=label, color=c, linewidth=2)
        if n > 1:
            ax.fill_between(grid, mean - std, mean + std, color=c, alpha=0.18)
    if baseline is not None and np.isfinite(baseline):
        ax.axhline(baseline, ls="--", color="black", alpha=0.6,
                   label=f"supervised upper bound ({baseline:.3f})")
    ax.set_xlabel("Labeled pool size")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", out_path)


def plot_dataset_curves(
    dataset: str,
    groups_for_dataset: dict[str, list[RunRecord]],
    baseline: float | None,
    out_dir: Path,
) -> None:
    """Accuracy, F1, and ECE learning-curve figures for one dataset."""
    out_dir.mkdir(parents=True, exist_ok=True)
    _curve_figure(groups_for_dataset, "accuracy", "Test accuracy",
                  f"{dataset} — accuracy vs label budget",
                  out_dir / f"{dataset}_accuracy.png", baseline=baseline)
    _curve_figure(groups_for_dataset, "f1", "Test macro-F1",
                  f"{dataset} — F1 vs label budget",
                  out_dir / f"{dataset}_f1.png")
    _curve_figure(groups_for_dataset, "ece", "ECE (lower = better calibrated)",
                  f"{dataset} — calibration (ECE) vs label budget",
                  out_dir / f"{dataset}_ece.png")


def plot_confusion_matrix(run: RunRecord, out_dir: Path) -> None:
    """Final-cycle confusion-matrix heatmap (row-normalized) for one representative run."""
    cms = sorted((run.run_dir / "confusion_matrices").glob("cycle_*.npy")) \
        if (run.run_dir / "confusion_matrices").exists() else []
    if not cms:
        return
    last = max(cms, key=lambda p: int(p.stem.split("_")[-1]))
    try:
        cm = np.load(last).astype(float)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not load %s: %s", last, exc)
        return
    row_sums = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(norm, cmap="viridis", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row-normalized")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{run.dataset_label} / {run.strategy} (seed {run.seed}) — final confusion")
    fig.tight_layout()
    out = out_dir / f"{run.dataset_label}_{run.strategy}_confusion.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", out)
