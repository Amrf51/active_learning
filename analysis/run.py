"""
run.py — CLI entry point for the AL analysis tooling.

    python -m analysis.run --exp-dir experiments --out analysis_out

Writes:
    {out}/figures/{dataset}_accuracy.png   (+ _f1, _ece, _confusion)
    {out}/summary.md                       (human-readable table + skipped/drift notes)
    {out}/summary.csv                      (one row per dataset x strategy)
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

from . import aggregate, plots, stats

logger = logging.getLogger("analysis.run")


def _fmt(x: float, nd: int = 3) -> str:
    return "—" if x != x else f"{x:.{nd}f}"  # x!=x catches NaN


def _write_csv(stat_list: list[stats.StrategyStat], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "dataset", "strategy", "n_seeds", "final_acc_mean", "final_acc_std",
            "auc_mean", "auc_std", "labels_to_90pct_baseline", "p_value_vs_random", "note",
        ])
        for s in stat_list:
            w.writerow([
                s.dataset, s.strategy, s.n_seeds, _fmt(s.final_acc_mean), _fmt(s.final_acc_std),
                _fmt(s.auc_mean), _fmt(s.auc_std),
                _fmt(s.labels_to_target_mean, 0),
                _fmt(s.p_value_vs_random) if s.p_value_vs_random is not None else "—",
                s.sig_note,
            ])


def _write_markdown(
    stat_list: list[stats.StrategyStat],
    discovery: dict,
    baselines: dict[str, float],
    path: Path,
) -> None:
    lines: list[str] = []
    lines.append("# Active Learning — strategy comparison summary\n")
    lines.append(
        f"Discovered **{discovery['total']}** runs: "
        f"**{discovery['al_runs']}** AL runs, **{discovery['baselines']}** supervised baselines, "
        f"**{len(discovery['skipped'])}** skipped (incomplete).\n"
    )
    if baselines:
        lines.append("**Supervised upper bounds:** " +
                     ", ".join(f"{d} = {a:.3f}" for d, a in sorted(baselines.items())) + "\n")

    datasets = sorted({s.dataset for s in stat_list})
    for ds in datasets:
        lines.append(f"\n## {ds}\n")
        lines.append("| Strategy | Seeds | Final acc (mean±std) | AUC (mean±std) | "
                     "Labels→90% baseline | vs random |")
        lines.append("|---|---|---|---|---|---|")
        for s in sorted([x for x in stat_list if x.dataset == ds], key=lambda x: x.strategy):
            lines.append(
                f"| {s.strategy} | {s.n_seeds} | "
                f"{_fmt(s.final_acc_mean)} ± {_fmt(s.final_acc_std)} | "
                f"{_fmt(s.auc_mean)} ± {_fmt(s.auc_std)} | "
                f"{_fmt(s.labels_to_target_mean, 0)} | {s.sig_note} |"
            )

    if discovery["skipped"]:
        lines.append("\n## Skipped runs (incomplete / aborted)\n")
        for run_dir, note in discovery["skipped"]:
            lines.append(f"- `{run_dir}` — {note}")

    lines.append(
        "\n---\n*AUC = mean test accuracy integrated over the label budget (label "
        "efficiency). Significance is a paired Wilcoxon signed-rank of per-seed AUC vs the "
        "random baseline; it needs ≥3 shared seeds.*\n"
    )
    path.write_text("\n".join(lines))
    logger.info("wrote %s", path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate + plot + score AL experiments.")
    parser.add_argument("--exp-dir", default="experiments", help="Experiments root")
    parser.add_argument("--out", default="analysis_out", help="Output directory")
    parser.add_argument("--target-fraction", type=float, default=0.9,
                        help="labels-to-target uses this fraction of the supervised baseline")
    parser.add_argument("--no-confusion", action="store_true", help="Skip confusion-matrix figures")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    out_dir = Path(args.out)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    records = aggregate.discover_runs(args.exp_dir)
    discovery = aggregate.summarize_discovery(records)
    logger.info("Discovered %d runs (%d AL, %d baseline, %d skipped)",
                discovery["total"], discovery["al_runs"], discovery["baselines"],
                len(discovery["skipped"]))
    for run_dir, note in discovery["skipped"]:
        logger.info("  SKIP %s — %s", run_dir, note)

    groups = aggregate.group_runs(records)   # logs config-drift warnings
    baselines = aggregate.baselines_by_dataset(records)

    if not groups:
        logger.warning("No complete AL runs found — nothing to plot. "
                       "(Run the matrix via scripts/run_matrix.py first.)")
    # figures, per dataset
    datasets: dict[str, dict[str, list]] = {}
    for (dataset, strategy), runs in groups.items():
        datasets.setdefault(dataset, {})[strategy] = runs
    for dataset, by_strategy in datasets.items():
        plots.plot_dataset_curves(dataset, by_strategy, baselines.get(dataset), fig_dir)
        if not args.no_confusion:
            # one representative run per strategy (lowest seed) for the confusion grid
            for strategy, runs in by_strategy.items():
                rep = min(runs, key=lambda r: r.seed)
                plots.plot_confusion_matrix(rep, fig_dir)

    stat_list = stats.compute_all(groups, baselines, target_fraction=args.target_fraction)
    _write_csv(stat_list, out_dir / "summary.csv")
    _write_markdown(stat_list, discovery, baselines, out_dir / "summary.md")

    logger.info("Done. Figures in %s, tables in %s", fig_dir, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
