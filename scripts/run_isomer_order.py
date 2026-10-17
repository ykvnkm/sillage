"""Can the models tell stereoisomers apart? From run_gnn.py's saved predictions (E-020).

    uv run python scripts/run_isomer_order.py        # seconds; results/gnn_isomers.csv

Macro AP barely sees stereo: a skeleton's isomers always share a split part, and only
a quarter of the molecules have their stereo written down. This asks the question
directly, on the test parts: of two isomers experts described differently, which one
does the model give the note to? 0.5 is a coin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sillage.paths import project_root
from sillage.research import train_v2
from sillage.research.dataset_v2 import build

SEEDS = (0, 1, 2, 3, 4)
VARIANTS = ("LR", "GB", "GNN x5 flat", "GNN x5", "GNN x5 flat + LR + GB", "GNN x5 + LR + GB")


def main() -> int:
    data = build()
    targets = data.targets(soft=True)
    interim = project_root() / "data" / "interim"
    rows = []
    for seed in SEEDS:
        saved = np.load(interim / f"gnn_test_predictions_seed{seed}.npz")
        test = saved["test"]
        groups = [data.groups[i] for i in test]
        for variant in VARIANTS:
            result = train_v2.isomer_order(saved[variant], targets[test], groups)
            rows.append({"variant": variant, "seed": seed, **result})

    table = pd.DataFrame(rows)
    out = project_root() / "results" / "gnn_isomers.csv"
    table.to_csv(out, index=False, float_format="%.4f")
    print(table.pivot(index="variant", columns="seed", values="isomer_order").round(3))
    print(f"\nwrote {out.relative_to(project_root())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
