#!/usr/bin/env python3
"""
run_experiment.py — headless single Active Learning run (no Streamlit).

Reuses the exact same backend the UI uses (core.worker.build_al_loop +
ActiveLearningLoop.run_all_cycles), so a CLI run produces an identical run directory
under experiments/ to one started from the dashboard.

Usage:
    python scripts/run_experiment.py --config configs/cars_al.yaml \
        --strategy entropy --seed 43

    # least_confidence sets uncertainty_method automatically:
    python scripts/run_experiment.py --config configs/cars_al.yaml \
        --strategy least_confidence --seed 42

    # quick local smoke test (CPU, no workers):
    python scripts/run_experiment.py --config configs/quick_test.yaml \
        --strategy random --seed 42 --device cpu --num-workers 0

The run is written to:
    {exp_dir}/{strategy}_seed{seed}/{YYYYMMDD_HHMM}_{run_id8}/
matching the controller's convention (core/controller.py).
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path

# Make the project root importable when run as `python scripts/run_experiment.py`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config  # noqa: E402
from core.worker import build_al_loop  # noqa: E402

logger = logging.getLogger("run_experiment")

# Strategies whose uncertainty_method must be pinned for get_strategy() to dispatch correctly.
# (ml/strategies.py: "least_confidence" and "entropy" map through uncertainty_method.)
_UNCERTAINTY_METHOD = {
    "least_confidence": "least_confidence",
    "entropy": "entropy",
}
_VALID_STRATEGIES = {"random", "entropy", "margin", "least_confidence"}


def _sanitize_name(name: str) -> str:
    """Filesystem-safe folder name — mirrors Controller._sanitize_experiment_name."""
    raw = str(name or "").strip()
    if not raw:
        return "al_experiment"
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", raw)
    safe = safe.strip().strip(".")
    return safe or "al_experiment"


def _seed_everything(seed: int) -> None:
    """Seed python/numpy/torch globally so different --seed values genuinely diverge.

    The data_manager/dataloader re-seed numpy locally for pool selection; this adds the
    torch (training) RNG so seeds vary the whole pipeline, not just the initial pool.
    """
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_run_dir(config) -> Path:
    """Replicate the controller's run-dir construction + config snapshot."""
    run_id = uuid.uuid4().hex
    safe_name = _sanitize_name(config.experiment.name)
    config.experiment.name = safe_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_dir = Path(config.experiment.exp_dir) / safe_name / f"{timestamp}_{run_id[:8]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config.save_to_file(run_dir / "config.yaml")
    return run_dir


def run_single(
    config_path: str,
    strategy: str,
    seed: int,
    *,
    name: str | None = None,
    device: str | None = None,
    num_workers: int | None = None,
    num_cycles: int | None = None,
) -> Path:
    """Load config + overrides, build the AL loop, run all cycles. Returns the run dir."""
    if strategy not in _VALID_STRATEGIES:
        raise ValueError(f"--strategy must be one of {sorted(_VALID_STRATEGIES)}, got {strategy!r}")

    overrides = {
        "active_learning.sampling_strategy": strategy,
        "experiment.seed": seed,
        "experiment.name": name or f"{strategy}_seed{seed}",
    }
    if strategy in _UNCERTAINTY_METHOD:
        overrides["active_learning.uncertainty_method"] = _UNCERTAINTY_METHOD[strategy]
    if device is not None:
        overrides["experiment.device"] = device
    if num_workers is not None:
        overrides["data.num_workers"] = num_workers
    if num_cycles is not None:
        overrides["active_learning.num_cycles"] = num_cycles

    config = load_config(config_path, overrides=overrides)
    _seed_everything(config.experiment.seed)

    run_dir = _build_run_dir(config)
    logger.info(
        "RUN start | strategy=%s seed=%s reset=%s loss=%s epochs(cap)=%s patience=%s -> %s",
        config.active_learning.sampling_strategy,
        config.experiment.seed,
        config.active_learning.reset_mode,
        config.training.loss_fn,
        config.training.epochs,
        config.training.early_stopping_patience,
        run_dir,
    )

    loop = build_al_loop(config, run_dir)
    loop.run_all_cycles()
    logger.info("RUN done  | %s", run_dir)
    return run_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one headless Active Learning experiment.")
    parser.add_argument("--config", required=True, help="Path to experiment YAML (e.g. configs/cars_al.yaml)")
    parser.add_argument("--strategy", required=True, choices=sorted(_VALID_STRATEGIES))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--name", default=None, help="Override experiment.name (default: {strategy}_seed{seed})")
    parser.add_argument("--device", default=None, help="Override device: cuda/cpu/auto")
    parser.add_argument("--num-workers", type=int, default=None, dest="num_workers",
                        help="Override data.num_workers (use 0 on macOS/Windows)")
    parser.add_argument("--num-cycles", type=int, default=None, dest="num_cycles",
                        help="Override active_learning.num_cycles (handy for smoke tests)")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    run_single(
        config_path=args.config,
        strategy=args.strategy,
        seed=args.seed,
        name=args.name,
        device=args.device,
        num_workers=args.num_workers,
        num_cycles=args.num_cycles,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
