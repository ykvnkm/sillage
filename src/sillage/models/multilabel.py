"""One binary classifier per label, fitted independently.

This is the "one-vs-rest" reduction: 138 separate yes/no questions instead of one
138-way problem. It ignores the correlations between descriptors that the EDA found
(`lily` and `muguet` co-occur 31x more often than chance), and that is precisely why
it belongs in a baseline -- a model that exploits label structure has to beat one
that does not, or the structure was not worth exploiting.

Written rather than taken from ``sklearn.multiclass.OneVsRestClassifier`` for one
reason: a label can have no positive examples in a training split. On the scaffold
split that is not hypothetical. sklearn raises; here such a label falls back to
predicting its (zero) prevalence, so the run completes and the metric layer reports
the label as unscored instead of the whole experiment dying.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, clone

EstimatorFactory = Callable[[], BaseEstimator]


class MultilabelClassifier:
    """Fits ``estimator_factory()`` once per label and stacks the probabilities."""

    def __init__(self, estimator_factory: EstimatorFactory, *, n_jobs: int = 1) -> None:
        """Serial by default, and that is faster here, not a compromise.

        Both baseline estimators already use every core: logistic regression through
        BLAS, histogram boosting through OpenMP. Wrapping them in process-level
        parallelism means ten workers each asking for ten cores. Measured on this
        dataset, ``n_jobs=-1`` made a full 138-label fit several times *slower* than
        ``n_jobs=1`` -- pure oversubscription.

        Raise it only for an estimator that is single-threaded internally, and then
        cap the inner threads (``threadpoolctl``) as well.
        """
        self.estimator_factory = estimator_factory
        self.n_jobs = n_jobs
        self.estimators_: list[Any] = []
        self.constant_scores_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> MultilabelClassifier:
        """Fit one classifier per column of *y*.

        Columns that are constant in the training data get no classifier at all:
        there is nothing to learn from a question whose answer never varies.
        """
        y = np.asarray(y)
        n_labels = y.shape[1]
        prevalence = y.mean(axis=0)
        degenerate = (prevalence == 0.0) | (prevalence == 1.0)

        fitted = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_one)(self.estimator_factory, x, y[:, index])
            for index in range(n_labels)
            if not degenerate[index]
        )

        self.estimators_ = []
        remaining = iter(fitted)
        for index in range(n_labels):
            self.estimators_.append(None if degenerate[index] else next(remaining))
        self.constant_scores_ = prevalence
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Probability of each label, shape ``(n_samples, n_labels)``."""
        if self.constant_scores_ is None:
            raise RuntimeError("Call fit before predict_proba")

        n_rows = x.shape[0]
        scores = np.empty((n_rows, len(self.estimators_)), dtype=np.float64)
        for index, estimator in enumerate(self.estimators_):
            if estimator is None:
                scores[:, index] = self.constant_scores_[index]
            else:
                scores[:, index] = estimator.predict_proba(x)[:, 1]
        return scores


def _fit_one(factory: EstimatorFactory, x: np.ndarray, y: np.ndarray) -> BaseEstimator:
    return clone(factory()).fit(x, y)


class PrevalenceBaseline:
    """Predicts each label's training frequency, identically for every molecule.

    The floor every real model must clear. It knows nothing about chemistry, which
    makes what it scores worth knowing: on this dataset it reaches micro AUC 0.78
    purely by ranking frequent descriptors above rare ones (journal entry E-004).
    """

    def __init__(self) -> None:
        self.prevalence_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> PrevalenceBaseline:
        self.prevalence_ = np.asarray(y).mean(axis=0)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.prevalence_ is None:
            raise RuntimeError("Call fit before predict_proba")
        return np.tile(self.prevalence_, (x.shape[0], 1))
