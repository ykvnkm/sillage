"""Metrics are checked against scikit-learn and against cases with known answers.

The equivalence tests are the important ones. Every comparison in phases 1 and 2
rests on these two numbers, so "our AP" must be *the* AP, not something close to it.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from sillage.metrics import (
    MultilabelReport,
    aggregate_reports,
    average_precision,
    evaluate,
    per_label_scores,
    roc_auc,
)


def random_case(seed: int, *, n: int = 200, prevalence: float = 0.2, tied: bool = False):
    """Random labels and scores; `tied` rounds scores hard to force many ties."""
    rng = np.random.default_rng(seed)
    labels = (rng.random(n) < prevalence).astype(np.int8)
    if labels.sum() in (0, n):  # keep both classes present
        labels[0], labels[-1] = 0, 1
    scores = rng.random(n) + 0.35 * labels  # weakly informative, like a real model
    return labels, np.round(scores, 1) if tied else scores


# --- equivalence with scikit-learn -------------------------------------------


@pytest.mark.parametrize("seed", range(25))
@pytest.mark.parametrize("tied", [False, True])
def test_average_precision_matches_sklearn(seed: int, tied: bool) -> None:
    labels, scores = random_case(seed, tied=tied)

    assert average_precision(labels, scores) == pytest.approx(
        average_precision_score(labels, scores), abs=1e-12
    )


@pytest.mark.parametrize("seed", range(25))
@pytest.mark.parametrize("tied", [False, True])
def test_roc_auc_matches_sklearn(seed: int, tied: bool) -> None:
    labels, scores = random_case(seed, tied=tied)

    assert roc_auc(labels, scores) == pytest.approx(roc_auc_score(labels, scores), abs=1e-12)


@pytest.mark.parametrize("prevalence", [0.01, 0.05, 0.5, 0.95])
def test_equivalence_holds_at_extreme_imbalance(prevalence: float) -> None:
    """The regime that actually matters: chamomile sits near 0.006."""
    labels, scores = random_case(0, n=2000, prevalence=prevalence)

    assert average_precision(labels, scores) == pytest.approx(
        average_precision_score(labels, scores), abs=1e-12
    )
    assert roc_auc(labels, scores) == pytest.approx(roc_auc_score(labels, scores), abs=1e-12)


# --- cases whose answer is known without a reference implementation ----------


def test_perfect_ranking_scores_one() -> None:
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])

    assert average_precision(labels, scores) == pytest.approx(1.0)
    assert roc_auc(labels, scores) == pytest.approx(1.0)


def test_exactly_wrong_ranking_scores_zero_auc() -> None:
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.9, 0.8, 0.2, 0.1])

    assert roc_auc(labels, scores) == pytest.approx(0.0)


def test_constant_scores_give_chance_performance() -> None:
    """All ties: nothing is ordered, so AUC is a coin flip and AP is the prevalence."""
    labels = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    scores = np.full(10, 0.5)

    assert roc_auc(labels, scores) == pytest.approx(0.5)
    assert average_precision(labels, scores) == pytest.approx(0.2)


def test_metrics_ignore_the_scale_of_the_scores() -> None:
    """Ranking metrics: only the induced order matters."""
    labels, scores = random_case(3)

    assert average_precision(labels, scores) == pytest.approx(
        average_precision(labels, scores * 1000 - 7)
    )
    assert roc_auc(labels, scores) == pytest.approx(roc_auc(labels, scores * 1000 - 7))


# --- undefined cases ---------------------------------------------------------


def test_average_precision_is_undefined_without_positives() -> None:
    """Not zero. There was nothing to find, so nothing was missed."""
    assert np.isnan(average_precision(np.zeros(5, dtype=np.int8), np.arange(5.0)))


@pytest.mark.parametrize("labels", [np.zeros(5, dtype=np.int8), np.ones(5, dtype=np.int8)])
def test_roc_auc_is_undefined_when_a_class_is_missing(labels: np.ndarray) -> None:
    assert np.isnan(roc_auc(labels, np.arange(5.0)))


def test_non_binary_labels_are_rejected() -> None:
    with pytest.raises(ValueError, match="binary"):
        average_precision(np.array([0, 1, 2]), np.array([0.1, 0.2, 0.3]))


def test_length_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="against"):
        roc_auc(np.array([0, 1]), np.array([0.1, 0.2, 0.3]))


# --- multilabel --------------------------------------------------------------

NAMES = ("common", "rare", "absent")


def multilabel_case() -> tuple[np.ndarray, np.ndarray]:
    """One frequent label the model ranks well, one rare label it cannot do at all.

    Scores share a single scale across labels, which micro-averaging silently
    requires. `test_micro_collapses_when_labels_are_not_on_one_scale` shows what
    happens when that assumption is broken.
    """
    rng = np.random.default_rng(0)
    n = 400
    labels = np.zeros((n, 3), dtype=np.int8)
    labels[: int(0.4 * n), 0] = 1
    labels[:8, 1] = 1
    rng.shuffle(labels)

    scores = rng.random((n, 3))
    scores[:, 0] += 0.6 * labels[:, 0]
    return labels, scores


def test_per_label_reports_the_baseline_next_to_the_metric() -> None:
    """AP alone is unreadable: 0.05 is good for a rare label and terrible for a common one."""
    labels, scores = multilabel_case()

    table = per_label_scores(labels, scores, NAMES)

    assert list(table.index) == list(NAMES)
    assert table.loc["common", "prevalence"] == pytest.approx(0.4)
    assert table.loc["common", "ap_lift"] == pytest.approx(
        table.loc["common", "average_precision"] / 0.4
    )


def test_a_label_without_positives_is_reported_as_unscored() -> None:
    labels, scores = multilabel_case()

    report = evaluate(labels, scores, NAMES)

    assert report.unscored_labels == ("absent",)
    assert report.n_labels_scored == 2
    assert report.n_labels_total == 3


def test_macro_average_skips_undefined_labels_instead_of_zeroing_them() -> None:
    """Filling the gap with zero would invent a measurement and drag macro down."""
    labels, scores = multilabel_case()

    report = evaluate(labels, scores, NAMES)
    scored = report.per_label["average_precision"].dropna()

    assert report.macro_average_precision == pytest.approx(scored.mean())
    assert report.macro_average_precision > scored.sum() / 3


def test_micro_follows_the_frequent_label_while_macro_feels_the_rare_failure() -> None:
    """The reason both averages are reported: they answer different questions.

    The model here is good at `common` and useless at `rare`. That failure halves
    the macro average and barely touches micro, because 160 of the 168 positives
    belong to `common`.
    """
    labels, scores = multilabel_case()

    report = evaluate(labels, scores, NAMES)
    common = report.per_label.loc["common", "average_precision"]

    assert abs(report.micro_average_precision - common) < abs(
        report.macro_average_precision - common
    )


def test_micro_collapses_when_labels_are_not_on_one_scale() -> None:
    """Micro pools scores across labels, so it measures cross-label calibration too.

    Shifting one label's scores by a constant changes no ranking *within* any label,
    so macro is untouched. Micro drops from roughly 0.78 to 0.15: the shifted label's
    negatives now outrank every genuine positive of the other one.
    """
    labels, scores = multilabel_case()
    shifted = scores.copy()
    shifted[:, 1] += 10.0

    before = evaluate(labels, scores, NAMES)
    after = evaluate(labels, shifted, NAMES)

    assert after.macro_average_precision == pytest.approx(before.macro_average_precision)
    assert after.micro_average_precision < before.micro_average_precision / 2


def test_summary_mentions_coverage() -> None:
    labels, scores = multilabel_case()

    text = evaluate(labels, scores, NAMES).summary()

    assert "2 of 3 labels" in text
    assert "absent" in text


def test_shape_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="shape"):
        evaluate(np.zeros((4, 3), dtype=np.int8), np.zeros((4, 2)), NAMES)


def test_name_count_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="names"):
        evaluate(np.zeros((4, 3), dtype=np.int8), np.zeros((4, 3)), ("only", "two"))


# --- aggregation across seeds ------------------------------------------------


def test_aggregate_reports_gives_mean_and_spread() -> None:
    labels, scores = multilabel_case()
    rng = np.random.default_rng(1)
    reports = [
        evaluate(labels, scores + rng.normal(0, 0.05, scores.shape), NAMES) for _ in range(5)
    ]

    table = aggregate_reports(reports)

    assert table.loc["macro_average_precision", "n_runs"] == 5
    assert table.loc["macro_average_precision", "std"] > 0


def test_aggregating_nothing_is_an_error() -> None:
    with pytest.raises(ValueError, match="Nothing to aggregate"):
        aggregate_reports([])


def test_a_single_run_has_no_spread() -> None:
    labels, scores = multilabel_case()

    table = aggregate_reports([evaluate(labels, scores, NAMES)])

    assert table.loc["macro_roc_auc", "std"] == 0.0


def test_report_is_immutable() -> None:
    labels, scores = multilabel_case()
    report = evaluate(labels, scores, NAMES)

    with pytest.raises((AttributeError, TypeError)):
        report.macro_roc_auc = 1.0  # type: ignore[misc]

    assert isinstance(report, MultilabelReport)
