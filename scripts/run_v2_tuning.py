"""Two levers never pulled on the baseline: regularisation strength, and an ensemble.

    uv run python scripts/run_v2_tuning.py          # ~12 minutes; results/v2_tuning.csv

On dataset v2 with soft, masked labels (variant C of E-019), five skeleton-grouped
splits, scored on unseen molecules against experts' consensus and each collection:

  LR C=c        logistic regression at regularisation c (the baseline used 0.1, untuned)
  LR best-val   the c with the best validation macro AP, chosen per split -- the honest one
  GB desc       histogram gradient boosting on the 216 descriptors
  LR + GB       mean of the two probabilities
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from sillage.features import sets as feature_sets
from sillage.models import MultilabelClassifier
from sillage.paths import project_root
from sillage.research import train_v2
from sillage.research.dataset_v2 import build

SEEDS = (0, 1, 2, 3, 4)
CS = (0.03, 0.1, 0.3, 1.0)


def main() -> int:
    warnings.filterwarnings("ignore")
    data = build()
    full = feature_sets.build("morgan-bin-r3+desc", data.molecules)
    desc = feature_sets.build("descriptors", data.molecules)
    targets = data.targets(soft=True)
    truths = {
        "consensus": data.consensus(),
        "leffingwell": data.leffingwell,
        "goodscents": data.goodscents,
    }

    rows = []
    for seed in SEEDS:
        split = data.split(seed=seed)
        by_c = {}
        for c in CS:
            model = fit(full, targets, split.train, c=c, seed=seed)
            by_c[c] = (
                model.predict_proba(full.values[split.val]),
                model.predict_proba(full.values[split.test]),
            )
            rows += _rows(f"LR C={c}", seed, by_c[c][1], truths, split.test)

        val_scores = {
            c: train_v2.score(v, targets[split.val])["macro_ap"] for c, (v, _) in by_c.items()
        }
        best = max(val_scores, key=val_scores.get)
        lr_test = by_c[best][1]
        rows += _rows("LR best-val", seed, lr_test, truths, split.test, chosen=best)

        gb = fit_boosting(desc, targets, split.train, seed=seed)
        gb_test = gb.predict_proba(desc.values[split.test])
        rows += _rows("GB desc", seed, gb_test, truths, split.test)
        rows += _rows(
            "LR best-val + GB", seed, (lr_test + gb_test) / 2, truths, split.test, chosen=best
        )
        print(f"seed {seed}: best C on validation = {best}", flush=True)

    table = pd.DataFrame(rows)
    out = project_root() / "results" / "v2_tuning.csv"
    table.to_csv(out, index=False, float_format="%.4f")
    print(table.groupby(["truth", "variant"]).macro_ap.agg(["mean", "std"]).round(4).to_string())
    print(f"\nwrote {out.relative_to(project_root())}")
    return 0


def fit(matrix, targets, rows, *, c: float, seed: int):
    return train_v2.fit(matrix.values, targets, rows, matrix.names, seed=seed, c=c)


def fit_boosting(matrix, targets, rows, *, seed: int):
    """Per-descriptor histogram boosting with the same missing/soft label handling."""
    estimators, constant = [], np.zeros(targets.shape[1])
    for j in range(targets.shape[1]):
        y = targets[rows, j]
        known = ~np.isnan(y)
        x, y = matrix.values[rows][known], y[known]
        constant[j] = float(y.mean()) if y.size else 0.0
        if not ((y > 0).any() and (y < 1).any()):
            estimators.append(None)
            continue
        hard = (y == 0) | (y == 1)
        x_all = np.vstack([x[hard], x[~hard], x[~hard]])
        y_all = np.concatenate([y[hard], np.ones((~hard).sum()), np.zeros((~hard).sum())])
        w_all = np.concatenate([np.ones(hard.sum()), y[~hard], 1 - y[~hard]])
        model = HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.1, random_state=seed, early_stopping=False
        )
        estimators.append(model.fit(x_all, y_all.astype(int), sample_weight=w_all))
    classifier = MultilabelClassifier(lambda: None)
    classifier.estimators_ = estimators
    classifier.constant_scores_ = constant
    return classifier


def _rows(name, seed, predicted, truths, test, chosen=None) -> list[dict]:
    out = []
    for truth_name, truth in truths.items():
        result = train_v2.score(predicted, truth[test])
        out.append(
            {"variant": name, "seed": seed, "truth": truth_name, "chosen_c": chosen, **result}
        )
    return out


if __name__ == "__main__":
    raise SystemExit(main())
