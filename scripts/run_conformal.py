"""What does a guarantee cost? Conformal note sets from the ensemble's scores.

    uv run python scripts/run_conformal.py     # ~1 min; needs the cached training jobs

A band like "likely" says how often a named note is right. It says nothing about the
notes left unnamed. Conformal prediction answers the other question: which notes must
be named so that a note the molecule really has is missed at most a tenth of the time?

Per note, split conformal on its positives: take the out-of-fold scores of the
learning molecules that have the note, and put the threshold at the score that all
but a share *alpha* of them exceed (with the finite-sample correction). A new molecule
gets every note whose score clears that note's threshold. For a note the molecule has,
the chance it is in the set is then at least 1 - alpha -- whatever the model's quality.
Quality decides the price: how many notes the set has to hold.

Checked on the test molecules with the shipped model. Journal E-024.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from sillage.models.artifact import OdorModel
from sillage.paths import model_dir, project_root
from sillage.research.dataset_v2 import build
from sillage.research.train_model_v2 import FOLDS
from sillage.research.train_model_v3 import _folds

SEED = 0
ALPHAS = (0.1, 0.2, 0.3)
MIN_CALIBRATION = 20
"""Positives a note needs for its threshold to mean anything: with fewer than 1/alpha
of them the corrected quantile is the minimum, and the note would always be named."""
MIN_TEST = 5


def thresholds(scores: np.ndarray, positive: np.ndarray, alpha: float) -> np.ndarray:
    """Per note, the score to clear; 0 (always named) where positives are too few."""
    out = np.zeros(scores.shape[1])
    for j in range(scores.shape[1]):
        own = np.sort(scores[positive[:, j], j])
        if own.size < MIN_CALIBRATION:
            continue
        # The floor(alpha * (n + 1))-th smallest positive score: at most that many of
        # n + 1 exchangeable positives fall strictly below it.
        rank = int(np.floor(alpha * (own.size + 1)))
        out[j] = own[rank - 1] if rank >= 1 else 0.0
    return out


def main() -> int:
    data = build()
    split = data.split(seed=SEED)
    experts = data.targets(soft=False)
    learn = np.concatenate([split.train, split.val])
    folds = _folds(learn, data.groups, SEED)
    interim = project_root() / "data" / "interim"
    raw = np.zeros((learn.size, experts.shape[1]))
    for k in range(FOLDS):
        raw[folds == k] = joblib.load(interim / f"ensemble_job_{k}_seed{SEED}.joblib")[1]

    model = OdorModel.load(model_dir())
    test = model.scorer().raw([data.molecules[i] for i in split.test])
    has = experts[split.test] == 1
    learn_has = experts[learn] == 1
    measured = (learn_has.sum(axis=0) >= MIN_CALIBRATION) & (has.sum(axis=0) >= MIN_TEST)

    rows = []
    for alpha in ALPHAS:
        named = test >= thresholds(raw, learn_has, alpha)[None, :]
        caught = (named & has)[:, measured].sum(axis=0) / has[:, measured].sum(axis=0)
        size = named[:, measured].sum(axis=1)
        rows.append(
            {
                "alpha": alpha,
                "promised_coverage": 1 - alpha,
                "notes_measured": int(measured.sum()),
                "coverage_mean": float(caught.mean()),
                "coverage_pooled": float((named & has)[:, measured].sum() / has[:, measured].sum()),
                "notes_below_promise_by_10pp": int((caught < 1 - alpha - 0.10).sum()),
                "set_size_median": float(np.median(size)),
                "set_size_p90": float(np.quantile(size, 0.9)),
                "true_notes_mean": float(has[:, measured].sum(axis=1).mean()),
            }
        )
    table = pd.DataFrame(rows)
    out = project_root() / "results" / f"conformal_{model.card.name}.csv"
    table.to_csv(out, index=False, float_format="%.4f")
    print(table.T.to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\nwrote {out.relative_to(project_root())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
