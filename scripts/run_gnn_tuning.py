"""The graph network's settings, chosen on validation only (journal E-020).

    uv sync --group gnn
    uv run python scripts/run_gnn_tuning.py          # ~35 minutes; results/gnn_tuning.csv

One change at a time from Chemprop's defaults, then the best three together, on the
validation part of splits 0 and 1. The test parts are never looked at here: they are
what scripts/run_gnn.py reports, and a setting picked by its test score would make that
report optimistic.
"""

from __future__ import annotations

import time
import warnings

import pandas as pd

from sillage.features.graphs import molecular_graph
from sillage.paths import project_root
from sillage.research import gnn
from sillage.research.dataset_v2 import build

SEEDS = (0, 1)
C = gnn.TrainingConfig
GRID = {
    "base": C(),
    "ep100": C(epochs=100, patience=25),
    "h600": C(hidden=600),
    "d3": C(depth=3),
    "d5": C(depth=5),
    "drop0.2": C(dropout=0.2),
    "drop0": C(dropout=0.0),
    "lr5e-4": C(learning_rate=5e-4, epochs=100, patience=25),
    "lr2e-3": C(learning_rate=2e-3),
    "bs32": C(batch_size=32),
    "h600+ep100": C(hidden=600, epochs=100, patience=25),
    "h600+lr2e-3": C(hidden=600, learning_rate=2e-3),
    "h600+lr2e-3+ep100": C(hidden=600, learning_rate=2e-3, epochs=100, patience=25),
}


def main() -> int:
    warnings.filterwarnings("ignore")
    data = build()
    graphs = [molecular_graph(m) for m in data.molecules]
    targets = data.targets(soft=True)
    experts = data.targets(soft=False)

    rows = []
    for seed in SEEDS:
        split = data.split(seed=seed)
        for name, config in GRID.items():
            started = time.time()
            trained = gnn.train(
                graphs, targets, split.train, split.val, experts[split.val],
                config=config, seed=seed,
            )  # fmt: skip
            rows.append(
                {
                    "variant": name,
                    "seed": seed,
                    "val_macro_ap": trained.best_val_ap,
                    "best_epoch": trained.best_epoch,
                    "seconds": round(time.time() - started),
                }
            )
            print(rows[-1], flush=True)

    table = pd.DataFrame(rows)
    out = project_root() / "results" / "gnn_tuning.csv"
    table.to_csv(out, index=False, float_format="%.4f")
    print(table.groupby("variant").val_macro_ap.mean().sort_values().round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
