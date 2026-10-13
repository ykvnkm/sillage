"""Does fixing the labels make a better model? Five variants on one split (journal E-019).

    uv run sillage-data fetch
    uv run python scripts/run_dataset_v2.py          # ~4 minutes; results/dataset_v2.csv

Every variant trains on the same skeleton-grouped training part of dataset v2 and is
scored on the same unseen molecules, against labels both expert collections agree on.
Each step adds one fix, so each row's gain over the previous one is that fix's worth:

  A  the merge's rules: "not asked" is "no", any yes is yes
  B  + "not asked" is missing, not "no"
  C  + disagreement is a soft 0.5
  D  + stereo-aware fingerprints
  E  the shipping recipe on OpenPOM's own merge, retrained without these test molecules
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from rdkit import Chem

from sillage.features import sets as feature_sets
from sillage.models import ChemicalScope
from sillage.paths import project_root
from sillage.research import train_v2
from sillage.research.baseline import Dataset
from sillage.research.dataset_v2 import build

FLAT = "morgan-bin-r3+desc"
CHIRAL = "morgan-bin-r3-chiral+desc"


SEEDS = (0, 1, 2, 3, 4)


def main() -> int:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    data = build()
    features = {
        FLAT: feature_sets.build(FLAT, data.molecules),
        CHIRAL: feature_sets.build(CHIRAL, data.molecules),
    }
    openpom = Dataset.load()
    openpom = openpom.restrict(openpom.in_scope(ChemicalScope()))
    openpom_groups = [Chem.MolToSmiles(m, isomericSmiles=False) for m in openpom.molecules]
    openpom_features = feature_sets.build(FLAT, openpom.molecules)
    variants = [
        ("A merge rules", FLAT, data.targets(soft=False, mask_unasked=False)),
        ("B + not asked is missing", FLAT, data.targets(soft=False)),
        ("C + disagreement is 0.5", FLAT, data.targets(soft=True)),
        ("D + stereo-aware features", CHIRAL, data.targets(soft=True)),
    ]
    # What each variant is scored against, on the test part: labels both collections
    # agree on, and each collection on its own (NaN where it was not asked).
    truths = {
        "consensus": data.consensus(),
        "goodscents": data.goodscents,
        "leffingwell": data.leffingwell,
    }

    rows = []
    for seed in SEEDS:
        split = data.split(seed=seed)
        test = split.test
        for name, feature_set, targets in variants:
            matrix = features[feature_set]
            model = train_v2.fit(matrix.values, targets, split.train, matrix.names, seed=seed)
            predicted = model.predict_proba(matrix.values[test])
            rows += _rows(name, seed, predicted, truths, test)

        held_out = {data.groups[i] for i in np.concatenate([split.val, test])}
        train_rows = np.flatnonzero([g not in held_out for g in openpom_groups])
        model = train_v2.fit(
            openpom_features.values,
            openpom.labels.astype(float),
            train_rows,
            openpom_features.names,
            seed=seed,
        )
        predicted = model.predict_proba(features[FLAT].values[test])
        rows += _rows("E OpenPOM merge", seed, predicted, truths, test)
        print(f"seed {seed} done", flush=True)

    table = pd.DataFrame(rows)
    out = project_root() / "results" / "dataset_v2.csv"
    table.to_csv(out, index=False, float_format="%.4f")
    summary = (
        table.groupby(["truth", "variant"])
        .agg(
            macro_ap=("macro_ap", "mean"),
            sd=("macro_ap", "std"),
            macro_auc=("macro_auc", "mean"),
            descriptors=("descriptors", "median"),
        )
        .round(4)
    )
    print(summary.to_string())
    print(f"\nwrote {out.relative_to(project_root())}")
    return 0


def _rows(name, seed, predicted, truths, test) -> list[dict]:
    out = []
    for truth_name, truth in truths.items():
        result = train_v2.score(predicted, truth[test])
        out.append({"variant": name, "seed": seed, "truth": truth_name, **result})
    return out


if __name__ == "__main__":
    raise SystemExit(main())
