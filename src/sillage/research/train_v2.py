"""Training with missing and soft labels, and scoring on experts' consensus (journal E-019).

Each descriptor's classifier sees only the molecules whose collections could express
that descriptor. A disputed label (one collection yes, the other no) becomes two
half-weight rows, one positive and one negative -- logistic regression's way of
saying "probability one half". The result is an ordinary
:class:`~sillage.models.MultilabelClassifier`, so it compiles, calibrates and ships
exactly like the baseline.
"""

from __future__ import annotations

from functools import partial

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
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


def fit_boosting(matrix, targets: np.ndarray, rows: np.ndarray, *, seed: int = 0):
    """Per-descriptor histogram boosting, with the same missing and soft label handling.

    Takes a :class:`~sillage.features.matrix.FeatureMatrix`; meant for the 216
    descriptors, where trees see thresholds a linear model cannot (journal E-019).
    """
    # A partial of the class, not a lambda: the classifier is saved with the model.
    factory = partial(
        HistGradientBoostingClassifier,
        max_iter=150,
        learning_rate=0.1,
        random_state=seed,
        early_stopping=False,
    )
    classifier = MultilabelClassifier(factory)
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
        model = factory()
        estimators.append(model.fit(x_all, y_all.astype(int), sample_weight=w_all))
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


def isomer_order(predicted: np.ndarray, targets: np.ndarray, groups) -> dict[str, float]:
    """How often the isomer experts gave a note is ranked above the one they did not.

    Over every pair of molecules sharing a skeleton (*groups*, aligned with the rows)
    and every note whose targets differ by at least one half. 0.5 is a coin -- and
    exactly what any model blind to stereo scores, since it cannot tell them apart.
    """
    by_group: dict[str, list[int]] = {}
    for row, group in enumerate(groups):
        by_group.setdefault(group, []).append(row)
    first, second, note = [], [], []
    for rows in by_group.values():
        for a in rows:
            for b in rows:
                gap = np.nan_to_num(targets[a] - targets[b], nan=0.0)
                for j in np.flatnonzero(gap >= 0.5):
                    first.append(a)
                    second.append(b)
                    note.append(j)
    if not note:
        return {"isomer_order": float("nan"), "isomer_pairs": 0}
    gap = predicted[first, note] - predicted[second, note]
    right = np.where(gap > 1e-6, 1.0, np.where(gap < -1e-6, 0.0, 0.5))
    return {"isomer_order": float(right.mean()), "isomer_pairs": len(note)}
