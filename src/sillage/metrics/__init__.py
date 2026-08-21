"""Ranking metrics for multilabel odour prediction.

Accuracy is forbidden in this project and the reason is arithmetic, not taste: the
label matrix is 3.5% ones, so answering "no" to all 138 questions scores 96.5% and
knows nothing. What matters is the *order* the model puts molecules in, which is
what average precision and ROC-AUC measure.
"""

from sillage.metrics.multilabel import (
    COLUMNS,
    MultilabelReport,
    aggregate_reports,
    evaluate,
    per_label_scores,
)
from sillage.metrics.ranking import average_precision, roc_auc

__all__ = [
    "COLUMNS",
    "MultilabelReport",
    "aggregate_reports",
    "average_precision",
    "evaluate",
    "per_label_scores",
    "roc_auc",
]
