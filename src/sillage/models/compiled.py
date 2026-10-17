"""Folding 138 scaled logistic regressions into one matrix product.

The shipping model is 138 independent scikit-learn pipelines, each a column-wise
scaler followed by a logistic regression. Calling them one by one costs about 114 ms
per molecule, almost all of it per-call overhead: the features themselves take 2 ms.

Both stages are linear, so they compose. For a feature scaled as
``z = (x - mean) / scale``, a regression ``w @ z + b`` equals

    (w / scale) @ x + (b - w @ (mean / scale))

which is a plain linear function of the unscaled input. Doing that once per label
and stacking the results gives a single ``(n_features, n_labels)`` matrix: one
product instead of 138 pipelines, with the same numbers to floating-point precision.
``tests/test_compiled.py`` holds it to that.

This is an inference-time view of an already fitted model, not a different model.
The artifact on disk still stores the original pipelines -- they are what training
produced and what the metrics were measured on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, StandardScaler

from sillage.models.multilabel import MultilabelClassifier


class NotCompilable(TypeError):
    """The classifier contains something that is not a linear map."""


@dataclass(frozen=True, slots=True)
class LinearMultilabel:
    """``sigmoid(x @ weights + bias)`` per label; constant labels pass a fixed score."""

    weights: np.ndarray
    """Shape ``(n_features, n_labels)``. Zero columns for constant labels."""

    bias: np.ndarray
    constant: np.ndarray
    """Boolean per label: True where training had no variation and no classifier."""

    constant_scores: np.ndarray

    @property
    def n_features_in_(self) -> int:
        return int(self.weights.shape[0])

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        logits = np.asarray(x, dtype=np.float64) @ self.weights + self.bias
        scores = 1.0 / (1.0 + np.exp(-logits))
        return np.where(self.constant, self.constant_scores, scores)

    @classmethod
    def from_classifier(cls, classifier: MultilabelClassifier) -> LinearMultilabel:
        """Fold every per-label pipeline. Raises :class:`NotCompilable` otherwise."""
        if classifier.constant_scores_ is None:
            raise NotCompilable("The classifier has not been fitted")

        width = _width(classifier)
        n_labels = len(classifier.estimators_)
        weights = np.zeros((width, n_labels))
        bias = np.zeros(n_labels)
        constant = np.array([e is None for e in classifier.estimators_])

        for index, estimator in enumerate(classifier.estimators_):
            if estimator is not None:
                weights[:, index], bias[index] = _fold(estimator, width)

        return cls(
            weights=weights,
            bias=bias,
            constant=constant,
            constant_scores=np.asarray(classifier.constant_scores_, dtype=np.float64),
        )


def _width(classifier: MultilabelClassifier) -> int:
    for estimator in classifier.estimators_:
        if estimator is not None:
            return int(estimator.n_features_in_)
    raise NotCompilable("No fitted estimators to compile")


def _fold(pipeline: object, width: int) -> tuple[np.ndarray, float]:
    """One label's pipeline as weights over the raw input columns, plus a bias."""
    if not isinstance(pipeline, Pipeline) or len(pipeline.steps) != 2:
        raise NotCompilable(f"Expected a two-step scaler + regression pipeline, got {pipeline!r}")
    scaler, regression = pipeline.steps[0][1], pipeline.steps[1][1]
    if not isinstance(regression, LogisticRegression) or regression.coef_.shape[0] != 1:
        raise NotCompilable(f"Expected a binary logistic regression, got {regression!r}")

    # Per raw input column: the shift and the divisor the scaler applies, and where
    # that column lands in the scaler's output (ColumnTransformer reorders by block).
    shift = np.zeros(width)
    divisor = np.ones(width)
    position = np.full(width, -1)

    if isinstance(scaler, ColumnTransformer):
        if scaler.remainder != "drop":
            raise NotCompilable("A ColumnTransformer passing columns through is not supported")
        offset = 0
        for _, transformer, columns in scaler.transformers_:
            if isinstance(transformer, str):  # "drop" for an empty block
                continue
            columns = np.asarray(columns, dtype=int)
            shift[columns], divisor[columns] = _affine(transformer)
            position[columns] = offset + np.arange(columns.size)
            offset += columns.size
    else:
        shift[:], divisor[:] = _affine(scaler)
        position[:] = np.arange(width)

    coef = regression.coef_[0]
    weights = np.zeros(width)
    used = position >= 0
    weights[used] = coef[position[used]] / divisor[used]
    bias = float(regression.intercept_[0] - np.sum(weights * shift))
    return weights, bias


def _affine(scaler: object) -> tuple[np.ndarray, np.ndarray]:
    """``(shift, divisor)`` such that the scaler computes ``(x - shift) / divisor``."""
    if isinstance(scaler, StandardScaler):
        shift = scaler.mean_ if scaler.with_mean else np.zeros_like(scaler.scale_)
        return shift, scaler.scale_
    if isinstance(scaler, MaxAbsScaler):
        return np.zeros_like(scaler.scale_), scaler.scale_
    raise NotCompilable(f"Scaler {type(scaler).__name__} is not a supported affine map")


@dataclass(frozen=True, slots=True)
class BoostedMultilabel:
    """Every label's boosted trees in flat arrays, walked for all trees at once.

    Scikit-learn's histogram boosting predicts one label at a time, and on a single
    molecule its per-call overhead dominates: 138 labels cost about 60 ms (journal
    E-020). The trees themselves are small -- at most 31 leaves -- so here every node
    of every tree sits in one set of arrays, and each step of the descent moves all
    20 000-odd trees one level down with a handful of NumPy gathers. Same numbers as
    scikit-learn to floating-point precision; ``tests/test_compiled.py`` holds it to that.
    """

    feature: np.ndarray
    threshold: np.ndarray
    missing_left: np.ndarray
    left: np.ndarray
    right: np.ndarray
    is_leaf: np.ndarray
    value: np.ndarray
    roots: np.ndarray
    """Global index of each tree's root."""

    tree_label: np.ndarray
    """Label each tree contributes to."""

    baseline: np.ndarray
    constant: np.ndarray
    constant_scores: np.ndarray
    depth: int

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        node = np.broadcast_to(self.roots, (x.shape[0], self.roots.size)).copy()
        rows = np.arange(x.shape[0])[:, None]
        for _ in range(self.depth):
            if self.is_leaf[node].all():
                break
            value = x[rows, self.feature[node]]
            go_left = np.where(
                np.isnan(value), self.missing_left[node], value <= self.threshold[node]
            )
            step = np.where(go_left, self.left[node], self.right[node])
            node = np.where(self.is_leaf[node], node, step)
        n_labels = self.baseline.size
        raw = np.zeros((x.shape[0], n_labels))
        for i in range(x.shape[0]):
            raw[i] = np.bincount(self.tree_label, self.value[node[i]], minlength=n_labels)
        scores = 1.0 / (1.0 + np.exp(-(raw + self.baseline)))
        return np.where(self.constant, self.constant_scores, scores)

    @classmethod
    def from_classifier(cls, classifier: MultilabelClassifier) -> BoostedMultilabel:
        """Flatten every per-label boosting model. Raises :class:`NotCompilable` otherwise."""
        if classifier.constant_scores_ is None:
            raise NotCompilable("The classifier has not been fitted")
        fields = {k: [] for k in ("feature", "threshold", "missing_left", "left", "right")}
        fields |= {"is_leaf": [], "value": []}
        roots, tree_label = [], []
        n_labels = len(classifier.estimators_)
        baseline = np.zeros(n_labels)
        constant = np.array([e is None for e in classifier.estimators_])
        offset, depth = 0, 0
        for label, estimator in enumerate(classifier.estimators_):
            if estimator is None:
                continue
            if not isinstance(estimator, HistGradientBoostingClassifier) or (
                estimator.n_trees_per_iteration_ != 1
            ):
                raise NotCompilable(f"Expected binary histogram boosting, got {estimator!r}")
            if getattr(estimator, "is_categorical_", None) is not None:
                raise NotCompilable("Categorical splits are not supported")
            baseline[label] = float(np.ravel(estimator._baseline_prediction)[0])
            for (tree,) in estimator._predictors:
                nodes = tree.nodes
                fields["feature"].append(nodes["feature_idx"].astype(np.int64))
                fields["threshold"].append(nodes["num_threshold"])
                fields["missing_left"].append(nodes["missing_go_to_left"].astype(bool))
                fields["left"].append(nodes["left"].astype(np.int64) + offset)
                fields["right"].append(nodes["right"].astype(np.int64) + offset)
                fields["is_leaf"].append(nodes["is_leaf"].astype(bool))
                fields["value"].append(nodes["value"])
                roots.append(offset)
                tree_label.append(label)
                depth = max(depth, int(nodes["depth"].max()))
                offset += len(nodes)
        if not roots:
            raise NotCompilable("No fitted estimators to compile")
        return cls(
            **{k: np.concatenate(v) for k, v in fields.items()},
            roots=np.asarray(roots),
            tree_label=np.asarray(tree_label),
            baseline=baseline,
            constant=constant,
            constant_scores=np.asarray(classifier.constant_scores_, dtype=np.float64),
            depth=depth,
        )
