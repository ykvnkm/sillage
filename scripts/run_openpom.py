"""OpenPOM's network against ours: same molecules to learn from, same to be tested on.

    uv run python scripts/run_openpom.py                 # 10 members x 62 epochs
    uv run python scripts/run_openpom.py --members 1     # one network
    uv run python scripts/run_openpom.py --bootstrap 1000  # + interval of the gap, ~6 min

The published 0.89 AUROC of OpenPOM is measured on its own merge of the data, its own
split and with unasked labels counted as "no" -- not comparable with our numbers. So
its network is trained here by its own recipe (scripts/openpom_fit.py) on the learning
part of dataset v2, split 0, and scored on the 570 test molecules by the very function
that fills the model card. Journal E-025.

What OpenPOM is given follows its own data preparation: flat labels, a note is present
if either collection of experts used it, absent otherwise.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
from rdkit import Chem

from sillage.features.graphs import molecular_graph
from sillage.models.artifact import OdorModel
from sillage.paths import model_dir, project_root
from sillage.research.dataset_v2 import build
from sillage.research.train_model_v2 import _metrics

SEED = 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--members", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=62)
    parser.add_argument("--bootstrap", type=int, default=0, help="resamples of test molecules")
    args = parser.parse_args()

    data = build()
    split = data.split(seed=SEED)
    learn = np.concatenate([split.train, split.val])
    smiles = np.array([Chem.MolToSmiles(m) for m in data.molecules])
    union = np.nan_to_num(data.targets(soft=True), nan=0.0) > 0

    interim = project_root() / "data" / "interim"
    handed = interim / f"openpom_split{SEED}.npz"
    predicted = interim / f"openpom_predictions_split{SEED}_m{args.members}_e{args.epochs}.npz"
    if not predicted.exists():
        np.savez(
            handed,
            train_smiles=smiles[learn],
            train_y=union[learn].astype(np.float32),
            test_smiles=smiles[split.test],
        )
        fit = project_root() / "scripts" / "openpom_fit.py"
        command = ["uv", "run", "--script", str(fit), str(handed), str(predicted)]
        subprocess.run([*command, str(args.members), str(args.epochs)], check=True)
    members = np.load(predicted)["members"]

    experts = data.targets(soft=False)[split.test]
    consensus = data.consensus()[split.test]
    model = OdorModel.load(model_dir())
    scorer = model.scorer()
    tested = [data.molecules[i] for i in split.test]
    ours = scorer.raw(tested)
    # Graph against graph: our five networks without the linear model and the boosting.
    graphs = np.stack([scorer.graph.predict(molecular_graph(m)) for m in tested])

    rows = [
        {"model": model.card.name, **_metrics(ours, experts, consensus)},
        {"model": "our graph networks alone", **_metrics(graphs, experts, consensus)},
        {
            "model": f"openpom, mean of {len(members)}",
            **_metrics(members.mean(axis=0), experts, consensus),
        },
    ]
    single = [_metrics(m, experts, consensus) for m in members]
    rows.append(
        {"model": "openpom, one network (mean over members)"}
        | {key: float(np.mean([s[key] for s in single])) for key in single[0]}
    )
    table = pd.DataFrame(rows)
    out = project_root() / "results" / "openpom.csv"
    table.to_csv(out, index=False, float_format="%.4f")
    print(table.T.to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"\nwrote {out.relative_to(project_root())}")

    if args.bootstrap:  # paired: both models are scored on the same resampled molecules
        keys = ("macro_average_precision", "macro_roc_auc", "consensus_macro_average_precision")
        rng = np.random.default_rng(SEED)
        theirs = members.mean(axis=0)
        gaps = []
        with warnings.catch_warnings():  # a resample can lose every positive of a rare note
            warnings.simplefilter("ignore")
            for _ in range(args.bootstrap):
                i = rng.integers(0, len(experts), len(experts))
                a = _metrics(ours[i], experts[i], consensus[i])
                b = _metrics(theirs[i], experts[i], consensus[i])
                gaps.append([a[k] - b[k] for k in keys])
        low, mid, high = np.quantile(gaps, [0.025, 0.5, 0.975], axis=0)
        for k, lo, m, hi in zip(keys, low, mid, high, strict=True):
            print(f"ours - openpom, {k}: {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
