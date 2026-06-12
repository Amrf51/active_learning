#!/usr/bin/env python3
"""
run_matrix.py — run the full strategy x seed matrix for one dataset config, sequentially.

The thesis matrix is:
    {cars_al, neu_al} x {random, entropy, margin, least_confidence} x {seed 42, 43, 44}
i.e. 12 runs per dataset config (24 total across both datasets).

Usage:
    # one dataset, all 4 strategies x 3 seeds (12 runs), sequentially:
    python scripts/run_matrix.py --config configs/cars_al.yaml

    # customise the grid:
    python scripts/run_matrix.py --config configs/neu_al.yaml \
        --strategies entropy random --seeds 42 43 44 45 46

    # local smoke run of the whole matrix shape (fast, CPU):
    python scripts/run_matrix.py --config configs/quick_test.yaml \
        --seeds 42 --device cpu --num-workers 0 --num-cycles 2

On a SLURM cluster you typically do NOT use this driver — submit each
(strategy, seed) combination as a separate array job calling run_experiment.py, so the
runs execute in parallel. This driver is the convenient single-machine path.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_experiment import _VALID_STRATEGIES, run_single  # noqa: E402

logger = logging.getLogger("run_matrix")

DEFAULT_STRATEGIES = ["random", "entropy", "margin", "least_confidence"]
DEFAULT_SEEDS = [42, 43, 44]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a strategy x seed AL matrix for one config.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--strategies", nargs="+", default=DEFAULT_STRATEGIES,
                        choices=sorted(_VALID_STRATEGIES))
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=None, dest="num_workers")
    parser.add_argument("--num-cycles", type=int, default=None, dest="num_cycles")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    combos = [(s, seed) for s in args.strategies for seed in args.seeds]
    logger.info("Matrix: %d runs = %d strategies x %d seeds | config=%s",
                len(combos), len(args.strategies), len(args.seeds), args.config)

    results: list[tuple[str, int, str, str]] = []  # (strategy, seed, status, detail)
    for i, (strategy, seed) in enumerate(combos, 1):
        logger.info("[%d/%d] strategy=%s seed=%s", i, len(combos), strategy, seed)
        t0 = time.perf_counter()
        try:
            run_dir = run_single(
                config_path=args.config,
                strategy=strategy,
                seed=seed,
                device=args.device,
                num_workers=args.num_workers,
                num_cycles=args.num_cycles,
            )
            dt = time.perf_counter() - t0
            results.append((strategy, seed, "OK", f"{run_dir} ({dt:.0f}s)"))
        except Exception as exc:  # keep the matrix going if one run dies
            dt = time.perf_counter() - t0
            logger.error("RUN FAILED strategy=%s seed=%s after %.0fs: %s", strategy, seed, dt, exc)
            logger.debug("%s", traceback.format_exc())
            results.append((strategy, seed, "FAIL", f"{exc} ({dt:.0f}s)"))

    n_ok = sum(1 for _, _, status, _ in results if status == "OK")
    logger.info("=" * 60)
    logger.info("MATRIX COMPLETE: %d/%d succeeded", n_ok, len(results))
    for strategy, seed, status, detail in results:
        logger.info("  [%-4s] %-16s seed=%s | %s", status, strategy, seed, detail)
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
