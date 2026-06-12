"""
stats.py — quantitative comparison of AL strategies.

Metrics:
  * normalized AUC of the accuracy-vs-labels curve ("area learning ratio") — a single
    scalar capturing label efficiency over the whole budget.
  * labels-to-target — labels needed to first reach X% of the supervised baseline.
  * paired significance vs random — Wilcoxon signed-rank across seeds on a per-seed scalar
    (default: AUC), since the shared seed fixes the init pool => strategy-vs-random is paired.

Everything degrades gracefully: with too few seeds the significance test reports
"insufficient seeds" rather than emitting a meaningless p-value.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from .aggregate import RunRecord, dedupe_by_seed

logger = logging.getLogger("analysis.stats")

MIN_SEEDS_FOR_SIGNIFICANCE = 3

# np.trapz (numpy <2) was renamed to np.trapezoid (numpy >=2). Support both.
_trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz")


def _finite(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def normalized_auc(labels: np.ndarray, values: np.ndarray) -> float:
    """Mean value under the curve over the observed label range (trapezoid / span).

    Result is in the same units as `values` (e.g. accuracy in [0,1]); it is the average
    accuracy you'd get integrating uniformly over the label budget. Higher = more efficient.
    """
    x, y = _finite(np.asarray(labels, float), np.asarray(values, float))
    if x.size < 2:
        return float(y[0]) if y.size else float("nan")
    order = np.argsort(x)
    x, y = x[order], y[order]
    span = x[-1] - x[0]
    if span <= 0:
        return float(np.mean(y))
    return float(_trapezoid(y, x) / span)


def labels_to_target(labels: np.ndarray, accuracy: np.ndarray, target: float) -> float:
    """First label count whose accuracy >= target. NaN if never reached."""
    x, y = _finite(np.asarray(labels, float), np.asarray(accuracy, float))
    hit = np.where(y >= target)[0]
    if hit.size == 0:
        return float("nan")
    return float(x[hit[0]])


@dataclass
class StrategyStat:
    dataset: str
    strategy: str
    n_seeds: int
    final_acc_mean: float
    final_acc_std: float
    auc_mean: float
    auc_std: float
    labels_to_target_mean: float
    target_acc: float
    per_seed_auc: dict[int, float] = field(default_factory=dict)
    per_seed_final: dict[int, float] = field(default_factory=dict)
    # filled in by compare_to_random:
    p_value_vs_random: float | None = None
    sig_note: str = ""


def strategy_stats(
    runs: list[RunRecord],
    baseline_acc: float | None,
    target_fraction: float = 0.9,
) -> StrategyStat:
    """Aggregate per-seed scalars for one (dataset, strategy) group."""
    runs = dedupe_by_seed(runs)
    target = (baseline_acc * target_fraction) if baseline_acc else float("inf")
    per_auc = {r.seed: normalized_auc(r.labels, r.accuracy) for r in runs}
    per_final = {r.seed: float(r.accuracy[-1]) for r in runs if r.accuracy.size}
    per_l2t = [labels_to_target(r.labels, r.accuracy, target) for r in runs]

    finals = np.array(list(per_final.values()), float)
    aucs = np.array(list(per_auc.values()), float)
    l2t = np.array(per_l2t, float)

    return StrategyStat(
        dataset=runs[0].dataset_label,
        strategy=runs[0].strategy,
        n_seeds=len(runs),
        final_acc_mean=float(np.nanmean(finals)) if finals.size else float("nan"),
        final_acc_std=float(np.nanstd(finals)) if finals.size else float("nan"),
        auc_mean=float(np.nanmean(aucs)) if aucs.size else float("nan"),
        auc_std=float(np.nanstd(aucs)) if aucs.size else float("nan"),
        labels_to_target_mean=float(np.nanmean(l2t)) if np.isfinite(l2t).any() else float("nan"),
        target_acc=target if np.isfinite(target) else float("nan"),
        per_seed_auc=per_auc,
        per_seed_final=per_final,
    )


def compare_to_random(stat: StrategyStat, random_stat: StrategyStat | None) -> StrategyStat:
    """Paired Wilcoxon signed-rank of per-seed AUC, strategy vs random, matched by seed."""
    if stat.strategy == "random":
        stat.sig_note = "(reference)"
        return stat
    if random_stat is None:
        stat.sig_note = "no random baseline for this dataset"
        return stat

    shared = sorted(set(stat.per_seed_auc) & set(random_stat.per_seed_auc))
    if len(shared) < MIN_SEEDS_FOR_SIGNIFICANCE:
        stat.sig_note = f"insufficient seeds (n={len(shared)}; need >= {MIN_SEEDS_FOR_SIGNIFICANCE})"
        return stat

    a = np.array([stat.per_seed_auc[s] for s in shared])
    b = np.array([random_stat.per_seed_auc[s] for s in shared])
    diffs = a - b
    if np.allclose(diffs, 0):
        stat.p_value_vs_random = 1.0
        stat.sig_note = "identical to random"
        return stat
    try:
        from scipy.stats import wilcoxon
        res = wilcoxon(a, b)
        stat.p_value_vs_random = float(res.pvalue)
        direction = "better" if np.mean(diffs) > 0 else "worse"
        stat.sig_note = f"{direction} than random (n={len(shared)} seeds, p={res.pvalue:.3f})"
    except Exception as exc:  # noqa: BLE001
        stat.sig_note = f"test failed: {exc}"
    return stat


def compute_all(
    groups: dict[tuple[str, str], list[RunRecord]],
    baselines: dict[str, float],
    target_fraction: float = 0.9,
) -> list[StrategyStat]:
    """Stats for every group, with vs-random significance filled in per dataset."""
    stats_by_key = {
        key: strategy_stats(runs, baselines.get(dataset), target_fraction)
        for (dataset, strategy), runs in groups.items()
        for key in [(dataset, strategy)]
    }
    # random reference per dataset
    randoms = {ds: st for (ds, strat), st in stats_by_key.items() if strat == "random"}
    out = []
    for (dataset, strategy), st in stats_by_key.items():
        out.append(compare_to_random(st, randoms.get(dataset)))
    out.sort(key=lambda s: (s.dataset, s.strategy))
    return out
