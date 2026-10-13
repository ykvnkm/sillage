"""Training with missing and soft labels, and scoring on experts' consensus (journal E-019).

Each descriptor's classifier sees only the molecules whose collections could express
that descriptor. A disputed label (one collection yes, the other no) becomes two
half-weight rows, one positive and one negative -- logistic regression's way of
saying "probability one half". The result is an ordinary
:class:`~sillage.models.MultilabelClassifier`, so it compiles, calibrates and ships
exactly like the baseline.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from sillage.models import MultilabelClassifier
from sillage.models.baselines import BlockScaledLogisticRegression

MIN_EVAL = 5
"""Positives and negatives a descriptor needs among scored molecules to be averaged."""


def fit(
    features: np.ndarray,
    targets: np.ndarray,
    rows: np.ndarray,
    names,
    *,
    seed: int = 0,
    c: float = 0.1,
):
    """One logistic regression per descriptor over *rows*, skipping NaN targets."""
    factory = BlockScaledLogisticRegression(feature_names=tuple(names), c=c, seed=seed)
    classifier = MultilabelClassifier(factory)
    estimators, constant = [], np.zeros(targets.shape[1])
    for j in range(targets.shape[1]):
        y = targets[rows, j]
        known = ~np.isnan(y)
        x, y = features[rows][known], y[known]
        constant[j] = float(y.mean()) if y.size else 0.0
        if not ((y > 0).any() and (y < 1).any()):
            estimators.append(None)
            continue
        # A soft target t becomes a positive row of weight t and a negative of 1 - t.
        hard = (y == 0) | (y == 1)
        x_all = np.vstack([x[hard], x[~hard], x[~hard]])
        y_all = np.concatenate([y[hard], np.ones((~hard).sum()), np.zeros((~hard).sum())])
        w_all = np.concatenate([np.ones(hard.sum()), y[~hard], 1 - y[~hard]])
        model = factory()
        model.fit(x_all, y_all.astype(int), model__sample_weight=w_all)
        estimators.append(model)
    classifier.estimators_ = estimators
    classifier.constant_scores_ = constant
    return classifier


def score(predicted: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Macro AP and ROC-AUC over descriptors with enough known positives and negatives.

    *truth* may hold NaN (unknown) or a soft 0.5 (experts split) -- neither is a
    verdict to score against, so both are left out of that descriptor.
    """
    aps, aucs = [], []
    for j in range(truth.shape[1]):
        known = (truth[:, j] == 0) | (truth[:, j] == 1)
        y = truth[known, j]
        if (y == 1).sum() >= MIN_EVAL and (y == 0).sum() >= MIN_EVAL:
            aps.append(average_precision_score(y, predicted[known, j]))
            aucs.append(roc_auc_score(y, predicted[known, j]))
    return {
        "macro_ap": float(np.mean(aps)),
        "macro_auc": float(np.mean(aucs)),
        "descriptors": len(aps),
    }
