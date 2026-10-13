"""Calibration must change what the numbers mean without changing their order.

The invariant matters more than it looks. Average precision and AUC are ranking
metrics, so a monotone recalibration cannot move them -- and if it does, the
calibrator is silently reordering predictions and invalidating every measurement
taken before it was introduced.
"""

from __future__ import annotations

import numpy as np
import pytest

from sillage.metrics import (
    average_precision,
    brier_score,
    calibration_at_threshold,
    expected_calibration_error,
    reliability_table,
    roc_auc,
)
from sillage.models.calibration import (
    MIN_POSITIVES,
    Confidence,
    PerLabelCalibrator,
    confidence_of,
)


def overconfident_case(
    n: int = 600, n_labels: int = 4, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Scores that rank well but promise too much, like the real model did.

    True probability is a gentle function of a latent variable; the "model" reports a
    sharpened version of it. Ranking is right, the numbers are overstated.
    """
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n, n_labels))
    true_probability = 1.0 / (1.0 + np.exp(-latent))
    labels = (rng.random((n, n_labels)) < true_probability).astype(np.int8)
    sharpened = 1.0 / (1.0 + np.exp(-2.5 * latent))
    return sharpened, labels


def fit(scores: np.ndarray, labels: np.ndarray) -> PerLabelCalibrator:
    return PerLabelCalibrator.fit(scores, labels, base_rates=labels.mean(axis=0))


# --- the invariant -----------------------------------------------------------


def test_per_label_ranking_is_preserved_exactly() -> None:
    """Both branches of transform are monotone per column, so AP and AUC cannot move."""
    scores, labels = overconfident_case()
    calibrator = fit(scores, labels)

    calibrated = calibrator.transform(scores)

    for index in range(labels.shape[1]):
        assert average_precision(labels[:, index], calibrated[:, index]) == pytest.approx(
            average_precision(labels[:, index], scores[:, index]), abs=1e-12
        )
        assert roc_auc(labels[:, index], calibrated[:, index]) == pytest.approx(
            roc_auc(labels[:, index], scores[:, index]), abs=1e-12
        )


def test_an_uncalibrated_label_passes_its_score_through_untouched() -> None:
    """Substituting a constant would destroy ranking the model genuinely has."""
    scores, labels = overconfident_case(n_labels=2)
    labels[:, 1] = 0
    labels[: MIN_POSITIVES - 1, 1] = 1  # too few positives to fit

    calibrator = fit(scores, labels)
    calibrated = calibrator.transform(scores)

    assert not calibrator.fitted[1]
    np.testing.assert_array_equal(calibrated[:, 1], scores[:, 1])


# --- does it actually calibrate ----------------------------------------------


def test_overconfidence_is_reduced() -> None:
    scores, labels = overconfident_case()
    calibrator = fit(scores, labels)

    before = expected_calibration_error(labels, scores, equal_mass=False)
    after = expected_calibration_error(labels, calibrator.transform(scores), equal_mass=False)

    assert after < before


def test_the_brier_score_improves() -> None:
    scores, labels = overconfident_case()
    calibrator = fit(scores, labels)

    assert brier_score(labels, calibrator.transform(scores)) < brier_score(labels, scores)


def test_confident_claims_become_truthful() -> None:
    """The headline failure: promising 0.94 and delivering 0.83."""
    scores, labels = overconfident_case(n=2000)
    calibrated = fit(scores, labels).transform(scores)

    _, promised_before, observed_before = calibration_at_threshold(labels, scores, 0.8)
    _, promised_after, observed_after = calibration_at_threshold(labels, calibrated, 0.8)

    assert abs(promised_after - observed_after) < abs(promised_before - observed_before)


# --- refusing to fit nonsense ------------------------------------------------


def test_a_label_with_too_few_positives_is_left_alone() -> None:
    """Measured: 3-5 positives produce promises of 0.772 against reality of 0.364."""
    scores = np.tile(np.linspace(0.01, 0.99, 100).reshape(-1, 1), (1, 1))
    labels = np.zeros((100, 1), dtype=np.int8)
    labels[-(MIN_POSITIVES - 1) :, 0] = 1

    assert not fit(scores, labels).fitted[0]


def test_a_label_with_enough_positives_is_fitted() -> None:
    scores = np.linspace(0.01, 0.99, 200).reshape(-1, 1)
    labels = np.zeros((200, 1), dtype=np.int8)
    labels[-50:, 0] = 1

    assert fit(scores, labels).fitted[0]


def test_an_inverted_label_is_refused_rather_than_flipped() -> None:
    """A negative slope would reverse the ranking the ablation measured."""
    scores = np.linspace(0.01, 0.99, 200).reshape(-1, 1)
    labels = np.zeros((200, 1), dtype=np.int8)
    labels[:50, 0] = 1  # high scores are the negatives

    calibrator = fit(scores, labels)

    assert not calibrator.fitted[0]
    np.testing.assert_array_equal(calibrator.transform(scores)[:, 0], scores[:, 0])


def test_a_constant_label_is_refused() -> None:
    scores = np.linspace(0.01, 0.99, 50).reshape(-1, 1)

    assert not fit(scores, np.ones((50, 1), dtype=np.int8)).fitted[0]


def test_scores_at_the_extremes_do_not_blow_up() -> None:
    """Raw scores reach exactly 0 and 1, whose log-odds are infinite."""
    scores = np.array([[0.0], [1.0], [0.5], [0.0], [1.0]] * 20, dtype=np.float64)
    labels = np.array([[0], [1], [1], [0], [1]] * 20, dtype=np.int8)

    calibrated = fit(scores, labels).transform(scores)

    assert np.isfinite(calibrated).all()


def test_shape_mismatches_are_rejected() -> None:
    with pytest.raises(ValueError, match="shape"):
        PerLabelCalibrator.fit(
            np.zeros((10, 3)), np.zeros((10, 2), dtype=np.int8), base_rates=np.zeros(3)
        )


def test_transform_rejects_the_wrong_width() -> None:
    scores, labels = overconfident_case(n_labels=3)
    calibrator = fit(scores, labels)

    with pytest.raises(ValueError, match="3 labels"):
        calibrator.transform(np.zeros((5, 7)))


# --- the wording the product shows -------------------------------------------


@pytest.mark.parametrize(
    ("probability", "expected"),
    [
        (0.99, Confidence.LIKELY),
        (0.85, Confidence.LIKELY),
        (0.70, Confidence.PROBABLE),
        (0.55, Confidence.PROBABLE),
        (0.40, Confidence.POSSIBLE),
        (0.20, Confidence.UNLIKELY),
        (0.02, Confidence.NEGLIGIBLE),
    ],
)
def test_bands_follow_the_measured_edges(probability: float, expected: Confidence) -> None:
    assert confidence_of(probability, calibrated=True) is expected


@pytest.mark.parametrize("probability", [0.01, 0.5, 0.99])
def test_an_uncalibrated_label_never_claims_a_probability(probability: float) -> None:
    """31 of 138 descriptors land here. Their score ranks; it does not measure."""
    assert confidence_of(probability, calibrated=False) is Confidence.UNVERIFIED


def test_there_is_no_band_above_likely() -> None:
    """Separating 0.90 from 0.97 would need far more than the 99 pairs available."""
    assert confidence_of(0.90, calibrated=True) is confidence_of(0.999, calibrated=True)


# --- the reliability table ---------------------------------------------------


def test_the_table_reports_the_gap_in_both_directions() -> None:
    scores, labels = overconfident_case()

    table = reliability_table(labels, scores, bins=5)

    assert {"pairs", "predicted", "observed", "gap"} <= set(table.columns)
    assert table["pairs"].sum() == labels.size


def test_neither_binning_scheme_surfaces_a_tiny_confident_tail() -> None:
    """Documents why thresholds are used instead of bins, and it is not a preference.

    Equal-width bins let the near-zero mass dominate the average. Equal-mass bins
    collapse, because those pairs share almost identical values and the quantile
    edges land on top of one another. Measured on the real model: pooled ECE came out
    at 0.0034 while the top of the range promised 0.94 and delivered 0.83.
    """
    probability = np.concatenate([np.full(950, 0.02), np.full(50, 0.95)])
    truth = np.concatenate([np.zeros(950), np.zeros(50)]).astype(np.int8)

    wide = reliability_table(truth, probability, equal_mass=False, bins=10)
    mass = reliability_table(truth, probability, equal_mass=True, bins=10)

    # Both collapse to very few bins and both report a small average error, although
    # fifty predictions of 0.95 were all wrong.
    assert len(wide) <= 2
    assert len(mass) <= 2
    assert expected_calibration_error(truth, probability, equal_mass=False) < 0.1

    # The threshold view states the failure plainly.
    pairs, promised, observed = calibration_at_threshold(truth, probability, 0.8)
    assert pairs == 50
    assert promised == pytest.approx(0.95)
    assert observed == 0.0


def test_a_threshold_nobody_reaches_returns_no_number() -> None:
    """Inventing a mean over an empty selection is how fake metrics appear."""
    pairs, promised, observed = calibration_at_threshold(
        np.zeros(10, dtype=np.int8), np.full(10, 0.1), 0.9
    )

    assert pairs == 0
    assert np.isnan(promised)
    assert np.isnan(observed)


def test_unknown_labels_are_not_read_as_negatives() -> None:
    """NaN means nobody was asked (dataset v2). Treating it as 0 would drag the
    calibrated probability down for every molecule the label was never asked about."""
    rng = np.random.default_rng(3)
    scores = rng.uniform(0.05, 0.95, size=(400, 1))
    labels = (rng.uniform(size=(400, 1)) < scores).astype(float)
    with_unknown = labels.copy()
    with_unknown[200:] = np.nan  # half the rows: never asked

    as_negative = np.where(np.isnan(with_unknown), 0.0, with_unknown)
    base = np.array([0.5])
    skipped = PerLabelCalibrator.fit(scores, with_unknown, base_rates=base).transform(scores)
    zeroed = PerLabelCalibrator.fit(scores, as_negative, base_rates=base).transform(scores)
    truth = PerLabelCalibrator.fit(scores[:200], labels[:200], base_rates=base).transform(scores)

    np.testing.assert_allclose(skipped, truth)
    assert zeroed.mean() < skipped.mean() - 0.1
