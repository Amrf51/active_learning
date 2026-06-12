"""
aggregate.py — discover, parse, and group AL runs from the experiments/ tree.

Runs are grouped by CONFIG FIELDS (dataset = basename(data.data_dir), strategy =
active_learning.sampling_strategy), NOT by directory name — so inconsistent run-folder
names need no renaming, and runs from different seeds aggregate naturally.

Incomplete/aborted runs are detected and skipped (logged, never silently dropped).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger("analysis.aggregate")

# A run is "complete enough" to analyze if it finished at least this fraction of its
# configured cycles. Below this it is treated as aborted and skipped.
MIN_COMPLETION_FRACTION = 0.5
# A supervised-baseline run is a 1-cycle full-pool run; recognised separately for the
# upper-bound line rather than treated as an AL curve.
BASELINE_MAX_CYCLES = 1


@dataclass
class RunRecord:
    run_dir: Path
    dataset: str                 # basename of data.data_dir, e.g. "train" -> normalized below
    dataset_label: str           # human label, e.g. "Stanford_Cars" / "Neu_Surface"
    strategy: str
    seed: int
    epochs_cap: int
    configured_cycles: int
    completed_cycles: int
    # per-cycle arrays (aligned, length == completed_cycles)
    labels: np.ndarray = field(default_factory=lambda: np.array([]))
    accuracy: np.ndarray = field(default_factory=lambda: np.array([]))
    f1: np.ndarray = field(default_factory=lambda: np.array([]))
    ece: np.ndarray = field(default_factory=lambda: np.array([]))
    epochs_trained: np.ndarray = field(default_factory=lambda: np.array([]))
    final_per_class_f1: dict[str, float] = field(default_factory=dict)
    is_complete: bool = True
    is_baseline: bool = False
    note: str = ""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not read %s: %s", path, exc)
        return {}


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not read %s: %s", path, exc)
        return {}


def _dataset_label(data_dir: str) -> str:
    """Map a data_dir path to a stable, human dataset label.

    e.g. 'data/raw/Stanford_Cars/car_data/train' -> 'Stanford_Cars'
         'data/raw/Neu_Surface/train/images'      -> 'Neu_Surface'
    Falls back to the leaf folder name.
    """
    parts = [p for p in Path(str(data_dir)).parts if p not in (".", "")]
    known = {"stanford_cars": "Stanford_Cars", "neu_surface": "Neu_Surface"}
    for p in parts:
        if p.lower() in known:
            return known[p.lower()]
    # else: drop a trailing generic leaf like train/images
    generic = {"train", "test", "val", "validation", "images", "raw", "data"}
    meaningful = [p for p in parts if p.lower() not in generic]
    return meaningful[-1] if meaningful else (parts[-1] if parts else "unknown")


def discover_runs(exp_dir: str | Path) -> list[RunRecord]:
    """Walk experiments/*/*/ and parse every run that has a config.yaml.

    Returns ALL runs (complete and incomplete) with is_complete flagged, so callers can
    report what was skipped.
    """
    root = Path(exp_dir)
    records: list[RunRecord] = []
    if not root.exists():
        logger.warning("Experiments dir does not exist: %s", root)
        return records

    for run_dir in sorted(root.glob("*/*")):
        if not run_dir.is_dir():
            continue
        cfg_path = run_dir / "config.yaml"
        if not cfg_path.exists():
            continue  # not a run folder

        cfg = _read_yaml(cfg_path)
        al = cfg.get("active_learning", {})
        data = cfg.get("data", {})
        training = cfg.get("training", {})
        experiment = cfg.get("experiment", {})

        data_dir = data.get("data_dir", "")
        strategy = al.get("sampling_strategy", "unknown")
        configured_cycles = int(al.get("num_cycles", 0) or 0)
        rec = RunRecord(
            run_dir=run_dir,
            dataset=Path(str(data_dir)).name or "unknown",
            dataset_label=_dataset_label(data_dir),
            strategy=str(strategy),
            seed=int(experiment.get("seed", -1) or -1),
            epochs_cap=int(training.get("epochs", 0) or 0),
            configured_cycles=configured_cycles,
            completed_cycles=0,
        )

        results = _read_json(run_dir / "al_cycle_results.json")
        cycles = results.get("cycles", []) if isinstance(results, dict) else []
        if not cycles:
            rec.is_complete = False
            rec.note = "no cycles in al_cycle_results.json"
            records.append(rec)
            continue

        cycles = [c for c in cycles if isinstance(c, dict)]
        cycles.sort(key=lambda c: c.get("labeled_pool_size", c.get("cycle", 0)))
        rec.completed_cycles = len(cycles)
        rec.labels = np.array([c.get("labeled_pool_size", np.nan) for c in cycles], dtype=float)
        rec.accuracy = np.array([c.get("test_accuracy", np.nan) for c in cycles], dtype=float)
        rec.f1 = np.array([c.get("test_f1", np.nan) for c in cycles], dtype=float)
        rec.ece = np.array([c.get("ece", np.nan) for c in cycles], dtype=float)
        rec.epochs_trained = np.array([c.get("epochs_trained", np.nan) for c in cycles], dtype=float)

        last_pc = cycles[-1].get("per_class_metrics") or {}
        rec.final_per_class_f1 = {
            str(k): float(v.get("f1", np.nan))
            for k, v in last_pc.items() if isinstance(v, dict)
        }

        # Classify baseline vs AL run, and completeness.
        rec.is_baseline = configured_cycles <= BASELINE_MAX_CYCLES
        if not rec.is_baseline and configured_cycles > 0:
            frac = rec.completed_cycles / configured_cycles
            if frac < MIN_COMPLETION_FRACTION:
                rec.is_complete = False
                rec.note = f"only {rec.completed_cycles}/{configured_cycles} cycles"

        records.append(rec)

    return records


def group_runs(records: list[RunRecord]) -> dict[tuple[str, str], list[RunRecord]]:
    """Group complete, non-baseline AL runs by (dataset_label, strategy) across seeds.

    Emits a warning when runs within a group disagree on epochs cap / configured cycles
    / initial pool — i.e. config drift that would make averaging them unsound.
    """
    groups: dict[tuple[str, str], list[RunRecord]] = {}
    for r in records:
        if r.is_baseline or not r.is_complete:
            continue
        groups.setdefault((r.dataset_label, r.strategy), []).append(r)

    for (dataset, strategy), runs in groups.items():
        caps = {r.epochs_cap for r in runs}
        cyc = {r.configured_cycles for r in runs}
        init = {int(r.labels[0]) if r.labels.size else -1 for r in runs}
        if len(caps) > 1 or len(cyc) > 1 or len(init) > 1:
            logger.warning(
                "Config drift in group (%s, %s): epochs_cap=%s cycles=%s init_pool=%s "
                "across seeds %s — runs may not be directly comparable.",
                dataset, strategy, sorted(caps), sorted(cyc), sorted(init),
                sorted(r.seed for r in runs),
            )
    return groups


def dedupe_by_seed(runs: list[RunRecord]) -> list[RunRecord]:
    """Keep one run per seed (the most-complete; tie-break: longest label reach).

    Two runs can share a seed when a config was re-run (e.g. Cars Random_Strategy vs
    Random_Strategy2). Without this, per-seed aggregation would silently drop one and
    plots would draw two same-seed curves as a misleading std band.
    """
    best: dict[int, RunRecord] = {}
    for r in runs:
        cur = best.get(r.seed)
        reach = float(r.labels[-1]) if r.labels.size else -1.0
        cur_reach = float(cur.labels[-1]) if (cur and cur.labels.size) else -1.0
        if cur is None or (r.completed_cycles, reach) > (cur.completed_cycles, cur_reach):
            best[r.seed] = r
    if len(best) < len(runs):
        logger.warning(
            "Group (%s, %s): %d run(s) share seeds; kept the most-complete per seed.",
            runs[0].dataset_label, runs[0].strategy, len(runs) - len(best),
        )
    return list(best.values())


def baselines_by_dataset(records: list[RunRecord]) -> dict[str, float]:
    """Best full-pool supervised accuracy per dataset, for the upper-bound line."""
    out: dict[str, float] = {}
    for r in records:
        if not r.is_baseline or r.accuracy.size == 0:
            continue
        acc = float(np.nanmax(r.accuracy))
        if r.dataset_label not in out or acc > out[r.dataset_label]:
            out[r.dataset_label] = acc
    return out


def summarize_discovery(records: list[RunRecord]) -> dict[str, Any]:
    """Counts + the list of skipped runs, for logging/reporting."""
    skipped = [r for r in records if not r.is_complete]
    baselines = [r for r in records if r.is_baseline and r.is_complete]
    al_runs = [r for r in records if r.is_complete and not r.is_baseline]
    return {
        "total": len(records),
        "al_runs": len(al_runs),
        "baselines": len(baselines),
        "skipped": [(str(r.run_dir), r.note) for r in skipped],
    }
