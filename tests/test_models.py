"""Behaviour of the one-vs-rest wrapper and the baseline it has to beat."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from sillage.features import parse_smiles
from sillage.models import (
    MultilabelClassifier,
    PrevalenceBaseline,
    gradient_boosting,
    logistic_regression,
)
from sillage.research import feature_sets
from sillage.research.baseline import RunSpec, grid, summarise


@pytest.fixture
def toy() -> tuple[np.ndarray, np.ndarray]:
    """Three labels: learnable, learnable, and never positive."""
    rng = np.random.default_rng(0)
    x = rng.random((120, 5))
    labels = np.zeros((120, 3), dtype=np.int8)
    labels[:, 0] = (x[:, 0] > 0.5).astype(np.int8)
    labels[:, 1] = (x[:, 1] + x[:, 2] > 1.0).astype(np.int8)
    return x, labels


def small_factory():
    return LogisticRegression(max_iter=200)


# --- one-vs-rest wrapper -----------------------------------------------------


def test_predict_proba_has_one_column_per_label(toy) -> None:
    x, labels = toy

    scores = MultilabelClassifier(small_factory, n_jobs=1).fit(x, labels).predict_proba(x)

    assert scores.shape == labels.shape


def test_probabilities_stay_in_range(toy) -> None:
    x, labels = toy

    scores = MultilabelClassifier(small_factory, n_jobs=1).fit(x, labels).predict_proba(x)

    assert scores.min() >= 0.0
    assert scores.max() <= 1.0


def test_a_label_with_no_positives_does_not_break_the_run(toy) -> None:
    """On the scaffold split this is not hypothetical; sklearn would raise here."""
    x, labels = toy

    model = MultilabelClassifier(small_factory, n_jobs=1).fit(x, labels)
    scores = model.predict_proba(x)

    assert model.estimators_[2] is None
    assert np.all(scores[:, 2] == 0.0)


def test_labels_are_fitted_independently(toy) -> None:
    """One-vs-rest by definition: dropping a label cannot change another's scores."""
    x, labels = toy

    full = MultilabelClassifier(small_factory, n_jobs=1).fit(x, labels).predict_proba(x)
    subset = MultilabelClassifier(small_factory, n_jobs=1).fit(x, labels[:, :1]).predict_proba(x)

    np.testing.assert_allclose(full[:, 0], subset[:, 0])


def test_the_model_learns_something(toy) -> None:
    """A sanity floor: on a label that is a threshold on one feature, it must work."""
    x, labels = toy

    scores = MultilabelClassifier(small_factory, n_jobs=1).fit(x, labels).predict_proba(x)

    assert scores[labels[:, 0] == 1, 0].mean() > scores[labels[:, 0] == 0, 0].mean()


def test_predicting_before_fitting_is_an_error(toy) -> None:
    x, _ = toy

    with pytest.raises(RuntimeError, match="fit"):
        MultilabelClassifier(small_factory).predict_proba(x)


# --- the floor ---------------------------------------------------------------


def test_prevalence_baseline_predicts_training_frequencies(toy) -> None:
    x, labels = toy

    scores = PrevalenceBaseline().fit(x, labels).predict_proba(x)

    np.testing.assert_allclose(scores[0], labels.mean(axis=0))
    assert len(np.unique(scores[:, 0])) == 1  # identical for every molecule


def test_prevalence_baseline_ignores_the_features(toy) -> None:
    """It has to: the whole point is that it knows nothing about chemistry."""
    x, labels = toy

    model = PrevalenceBaseline().fit(x, labels)

    np.testing.assert_allclose(model.predict_proba(x), model.predict_proba(x * 100))


# --- estimator construction --------------------------------------------------


def test_logistic_regression_scales_inside_the_pipeline() -> None:
    """Fitting the scaler outside would fit it on the test split too: leakage."""
    assert logistic_regression(("morgan_bit_0", "MolWt")).named_steps["scale"] is not None


def test_fingerprint_and_descriptor_blocks_get_different_scalers() -> None:
    """One scaler for both measures the scaler instead of the features (E-005)."""
    pipeline = logistic_regression(("morgan_bit_0", "morgan_bit_1", "MolWt", "TPSA"))

    blocks = {
        name: type(transformer).__name__
        for name, transformer, _ in pipeline.named_steps["scale"].transformers
    }

    assert blocks == {"fingerprints": "MaxAbsScaler", "descriptors": "StandardScaler"}


def test_a_matrix_of_only_descriptors_gets_one_block() -> None:
    pipeline = logistic_regression(("MolWt", "TPSA"))

    assert [name for name, _, _ in pipeline.named_steps["scale"].transformers] == ["descriptors"]


def test_boosting_needs_no_scaler() -> None:
    """Trees split on thresholds, so monotone rescaling changes nothing."""
    assert not hasattr(gradient_boosting(), "named_steps")


# --- feature set registry ----------------------------------------------------


def test_every_registered_feature_set_builds() -> None:
    molecules = parse_smiles(["CCO", "c1ccccc1", "O=Cc1ccc(O)cc1"])

    for name in feature_sets.FEATURE_SETS:
        matrix = feature_sets.build(name, molecules)
        assert matrix.shape[0] == 3
        assert matrix.shape[1] == len(matrix.names)


def test_unknown_feature_set_lists_the_known_ones() -> None:
    with pytest.raises(KeyError, match="descriptors"):
        feature_sets.build("does-not-exist", [])


def test_count_and_binary_fingerprints_differ() -> None:
    """Otherwise experiment E-001 would be comparing a thing with itself."""
    molecules = parse_smiles(["CCCCCCCCCC=O"])

    binary = feature_sets.build("morgan-bin-r2", molecules).values
    counts = feature_sets.build("morgan-cnt-r2", molecules).values

    assert counts.max() > binary.max()


# --- grid bookkeeping --------------------------------------------------------


def test_grid_is_the_cartesian_product() -> None:
    specs = grid(["logreg"], ["descriptors", "morgan-bin-r2"], ["stratified"], [0, 1])

    assert len(specs) == 4
    assert len(set(specs)) == 4


def test_summarise_reports_spread_across_seeds() -> None:
    results = pd.DataFrame(
        {
            "model": ["logreg"] * 3,
            "features": ["descriptors"] * 3,
            "split": ["stratified"] * 3,
            "seed": [0, 1, 2],
            "macro_ap": [0.10, 0.12, 0.14],
            "macro_auc": [0.8, 0.8, 0.8],
            "micro_ap": [0.3, 0.3, 0.3],
            "micro_auc": [0.9, 0.9, 0.9],
            "labels_scored": [138, 138, 138],
            "seconds": [1.0, 1.0, 1.0],
        }
    )

    summary = summarise(results)

    assert summary["macro_ap_mean"].iloc[0] == pytest.approx(0.12)
    assert summary["macro_ap_std"].iloc[0] > 0
    assert summary["n_seeds"].iloc[0] == 3


def test_run_spec_prints_readably() -> None:
    assert "logreg" in str(RunSpec("logreg", "descriptors", "stratified", 0))
