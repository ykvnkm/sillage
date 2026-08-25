"""Run the phase 1 baseline grid and write the results table.

    uv run python scripts/run_baseline.py --models logreg --seeds 0 1 2

Results land in results/ and are committed: they are small, and a metric nobody can
find later is a metric nobody can check.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from sillage.paths import project_root
from sillage.research import feature_sets
from sillage.research.baseline import MODELS, SPLIT_STRATEGIES, Dataset, grid, run_grid, summarise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["logreg"], choices=list(MODELS))
    parser.add_argument(
        "--features",
        nargs="+",
        default=list(feature_sets.FEATURE_SETS),
        choices=list(feature_sets.FEATURE_SETS),
    )
    parser.add_argument(
        "--splits", nargs="+", default=["stratified"], choices=list(SPLIT_STRATEGIES)
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--tag", default="grid", help="name of the output files")
    args = parser.parse_args()

    results_dir = project_root() / "results"
    results_dir.mkdir(exist_ok=True)

    data = Dataset.load()
    specs = grid(args.models, args.features, args.splits, args.seeds)
    print(f"{len(specs)} runs\n")

    results, reports = run_grid(specs, data=data)

    _write(results, results_dir / f"baseline_{args.tag}_runs.csv")
    summary = summarise(results)
    _write(summary.reset_index(), results_dir / f"baseline_{args.tag}_summary.csv")

    best_spec = max(reports, key=lambda spec: reports[spec].macro_average_precision)
    _write(
        reports[best_spec].per_label.reset_index(),
        results_dir / f"baseline_{args.tag}_per_label_best.csv",
    )

    print("\n" + summary.round(4).to_string())
    print(f"\nbest run: {best_spec}")
    print(reports[best_spec].summary())
    return 0


def _write(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False)
    print(f"wrote {path.relative_to(project_root())}")


if __name__ == "__main__":
    raise SystemExit(main())
