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
