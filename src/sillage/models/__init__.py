"""Baseline models for structure -> odour prediction."""

from sillage.models.baselines import (
    gradient_boosting,
    logistic_regression,
    logistic_regression_single_scaler,
)
from sillage.models.multilabel import MultilabelClassifier, PrevalenceBaseline

__all__ = [
    "MultilabelClassifier",
    "PrevalenceBaseline",
    "gradient_boosting",
    "logistic_regression",
    "logistic_regression_single_scaler",
]
