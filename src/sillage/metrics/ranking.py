"""Average precision and ROC-AUC, implemented rather than imported.

Both take a vector of true labels and a vector of scores, and both are *ranking*
metrics: only the order the scores induce matters, never their absolute value. A
model whose scores are all divided by a thousand gets identical numbers here.

Implemented by hand because these two numbers decide every comparison in the
project, and a metric nobody in the project can derive is a metric nobody can
defend. ``tests/test_metrics.py`` checks both against scikit-learn on thousands of
random inputs, including the tie-heavy cases where naive implementations drift.
"""

from __future__ import annotations

import numpy as np


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the precision-recall curve, summed as steps.

    Walk down the ranked list. Every time a positive example appears, recall grows;
    the precision achieved at that moment is weighted by how much recall it added.
    Formally ``AP = sum_n (R_n - R_{n-1}) * P_n``.

    The number has no fixed baseline: a random ranking scores roughly the fraction
    of positives in the data. Interpreting AP without that fraction alongside it is
    the single most common way to misread this metric -- which is why
    :func:`sillage.metrics.multilabel.per_label_scores` always reports both.

    Returns:
        NaN when there are no positive examples. The metric is undefined then, not
        zero: there was nothing to find, so nothing was missed.
    """
    labels, scores = _validated(y_true, y_score)
    n_positive = int(labels.sum())
    if n_positive == 0:
        return float("nan")

    true_positives, false_positives = _cumulative_counts(labels, scores)
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / n_positive

    recall_gained = np.diff(recall, prepend=0.0)
    return float(np.sum(recall_gained * precision))


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Probability that a random positive outranks a random negative.

    Computed from ranks rather than by integrating a curve. Sum the ranks of the
    positives, subtract the smallest sum they could possibly have, and divide by
    the number of positive-negative pairs. Ties contribute one half, which falls
    out of using averaged ranks. This is the Mann-Whitney U statistic, and it is
    the definition -- the area under the ROC curve is a consequence of it.

    Returns:
        NaN when either class is absent: with nothing to separate, separation is
        undefined.
    """
    labels, scores = _validated(y_true, y_score)
    n_positive = int(labels.sum())
    n_negative = labels.size - n_positive
    if n_positive == 0 or n_negative == 0:
        return float("nan")

    ranks = _average_ranks(scores)
    smallest_possible = n_positive * (n_positive + 1) / 2
    return float((ranks[labels].sum() - smallest_possible) / (n_positive * n_negative))


def _cumulative_counts(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Positives and negatives seen so far, at each distinct score threshold.

    Grouping by *distinct* score matters. Examples sharing a score cannot be
    ordered relative to one another, so a threshold either admits all of them or
    none. Treating tied examples as separate steps would let the arbitrary order of
    the input inflate the metric.
    """
    order = np.argsort(-scores, kind="mergesort")
    labels_by_rank = labels[order]
    scores_by_rank = scores[order]

    last_of_each_score = np.r_[np.flatnonzero(np.diff(scores_by_rank)), labels.size - 1]
    true_positives = np.cumsum(labels_by_rank)[last_of_each_score]
    false_positives = 1 + last_of_each_score - true_positives
    return true_positives.astype(float), false_positives.astype(float)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks from 1 upwards, with tied values sharing the average of their ranks."""
    order = np.argsort(values, kind="mergesort")
    position = np.empty(values.size, dtype=np.int64)
    position[order] = np.arange(values.size)

    sorted_values = values[order]
    starts_new_group = np.r_[True, sorted_values[1:] != sorted_values[:-1]]
    group_of = starts_new_group.cumsum()[position]
    group_boundaries = np.r_[np.flatnonzero(starts_new_group), values.size]

    return 0.5 * (group_boundaries[group_of] + group_boundaries[group_of - 1] + 1)


def _validated(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(y_true).ravel()
    scores = np.asarray(y_score, dtype=np.float64).ravel()
    if labels.size != scores.size:
        raise ValueError(f"{labels.size} labels against {scores.size} scores")
    if labels.size == 0:
        raise ValueError("Cannot score an empty set of examples")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("Labels must be binary; these metrics rank, they do not regress")
    return labels.astype(bool), scores
