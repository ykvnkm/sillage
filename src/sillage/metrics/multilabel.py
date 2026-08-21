"""Scoring 138 labels at once, and the two ways of averaging that follow.

**Macro** averages the per-label scores: ``chamomile`` with 31 examples weighs the
same as ``fruity`` with 1902. It answers "how does the model do on a descriptor,
picked at random".

**Micro** pools every (molecule, label) pair into one big binary problem. Frequent
labels dominate it because they contribute most of the positives. It answers "how
does the model do on a prediction, picked at random".

Neither is the right one. They are different questions, they diverge sharply on
imbalanced data, and a gap between them is itself a finding: macro far below micro
means the model lives off the common descriptors.

Micro carries an assumption strong enough to be a warning: pooling requires scores
to be comparable *across* labels. Measured on the test fixture, adding a constant
to one label's scores -- which changes no ranking within any label, and leaves macro
bit-for-bit identical -- drops micro from 0.78 to 0.15. Micro therefore measures
cross-label calibration as much as it measures ranking, and a per-label calibration
step (phase 2) will move it without the model having changed at all.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sillage.metrics.ranking import average_precision, roc_auc

COLUMNS = ("n_positive", "prevalence", "average_precision", "ap_lift", "roc_auc")


def per_label_scores(
    y_true: np.ndarray,
    y_score: np.ndarray,
    names: Sequence[str],
) -> pd.DataFrame:
    """Score every label separately.

    Alongside each metric come the two numbers needed to read it: how many positive
    examples the label had, and what a random ranking would have scored. ``ap_lift``
    is average precision divided by that random baseline -- 1.0 means the model did
    exactly as well as chance.
    """
    labels, scores = _validated(y_true, y_score, names)
    n_rows = labels.shape[0]

    rows = []
    for index, name in enumerate(names):
        column = labels[:, index]
        n_positive = int(column.sum())
        prevalence = n_positive / n_rows
        ap = average_precision(column, scores[:, index])
        rows.append(
            {
                "label": name,
                "n_positive": n_positive,
                "prevalence": prevalence,
                "average_precision": ap,
                "ap_lift": ap / prevalence if prevalence > 0 else float("nan"),
                "roc_auc": roc_auc(column, scores[:, index]),
            }
        )

    return pd.DataFrame(rows).set_index("label")[list(COLUMNS)]


@dataclass(frozen=True, slots=True)
class MultilabelReport:
    """Per-label scores plus both aggregations, and an honest account of coverage."""

    per_label: pd.DataFrame
    macro_average_precision: float
    macro_roc_auc: float
    micro_average_precision: float
    micro_roc_auc: float
    unscored_labels: tuple[str, ...]
    """Labels with no positive examples here. Their metrics are undefined and are
    left out of the macro average rather than filled with zero, which would be a
    measurement nobody made."""

    @property
    def n_labels_total(self) -> int:
        return len(self.per_label)

    @property
    def n_labels_scored(self) -> int:
        return self.n_labels_total - len(self.unscored_labels)

    def summary(self) -> str:
        coverage = f"{self.n_labels_scored} of {self.n_labels_total} labels"
        lines = [
            f"macro AP  {self.macro_average_precision:.4f}   (over {coverage})",
            f"macro AUC {self.macro_roc_auc:.4f}",
            f"micro AP  {self.micro_average_precision:.4f}",
            f"micro AUC {self.micro_roc_auc:.4f}",
        ]
        if self.unscored_labels:
            listed = ", ".join(self.unscored_labels[:8])
            more = (
                f" and {len(self.unscored_labels) - 8} more"
                if len(self.unscored_labels) > 8
                else ""
            )
            lines.append(f"not scored: {listed}{more}")
        return "\n".join(lines)


def evaluate(
    y_true: np.ndarray,
    y_score: np.ndarray,
    names: Sequence[str],
) -> MultilabelReport:
    """Full multilabel evaluation: per label, macro and micro."""
    labels, scores = _validated(y_true, y_score, names)
    per_label = per_label_scores(labels, scores, names)

    unscored = tuple(per_label.index[per_label["n_positive"] == 0])

    return MultilabelReport(
        per_label=per_label,
        macro_average_precision=float(per_label["average_precision"].mean(skipna=True)),
        macro_roc_auc=float(per_label["roc_auc"].mean(skipna=True)),
        micro_average_precision=average_precision(labels.ravel(), scores.ravel()),
        micro_roc_auc=roc_auc(labels.ravel(), scores.ravel()),
        unscored_labels=unscored,
    )


def aggregate_reports(reports: Sequence[MultilabelReport]) -> pd.DataFrame:
    """Mean and spread of the aggregate metrics across several runs.

    With three positive examples of ``chamomile`` in the test part, a single number
    is a lottery ticket, not a measurement. Any claim that one model beats another
    has to survive the spread across seeds.
    """
    if not reports:
        raise ValueError("Nothing to aggregate")

    fields = {
        "macro_average_precision": [r.macro_average_precision for r in reports],
        "macro_roc_auc": [r.macro_roc_auc for r in reports],
        "micro_average_precision": [r.micro_average_precision for r in reports],
        "micro_roc_auc": [r.micro_roc_auc for r in reports],
    }
    return pd.DataFrame(
        {
            "mean": {name: float(np.mean(values)) for name, values in fields.items()},
            "std": {
                name: float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                for name, values in fields.items()
            },
            "n_runs": {name: len(reports) for name in fields},
        }
    )


def _validated(
    y_true: np.ndarray, y_score: np.ndarray, names: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(y_true)
    scores = np.asarray(y_score, dtype=np.float64)
    if labels.shape != scores.shape:
        raise ValueError(f"Labels have shape {labels.shape}, scores {scores.shape}")
    if labels.ndim != 2:
        raise ValueError(f"Expected a 2-D matrix of labels, got {labels.ndim}-D")
    if labels.shape[1] != len(names):
        raise ValueError(f"{labels.shape[1]} label columns but {len(names)} names")
    return labels, scores
