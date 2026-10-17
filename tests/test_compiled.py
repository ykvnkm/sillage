"""The compiled model must be the same model, only faster.

Measured on the shipping artifact: largest difference 1.5e-6 over 4777 molecules x
138 labels, and not one of 659,226 confidence words changed (journal E-013). These
tests hold every supported pipeline shape to that standard.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sillage.models import (
    BlockScaledLogisticRegression,
    MultilabelClassifier,
    gradient_boosting,
    logistic_regression_single_scaler,
)
from sillage.models.artifact import OdorModel
from sillage.models.compiled import LinearMultilabel, NotCompilable

NAMES = ("morgan_bit_0", "morgan_bit_1", "morgan_bit_2", "MolWt", "TPSA", "LogP")


@pytest.fixture
def data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    x = np.hstack([rng.integers(0, 2, (200, 3)), rng.normal(150, 60, (200, 3))])
    y = np.zeros((200, 4), dtype=np.int8)
    y[:, 0] = x[:, 0].astype(np.int8)
    y[:, 1] = (x[:, 3] > 150).astype(np.int8)
    y[:, 2] = ((x[:, 1] + (x[:, 4] > 160)) > 1).astype(np.int8)
    # Column 3 is never positive: no classifier, a constant score.
    return x, y


@pytest.mark.parametrize(
    "factory",
    [
        BlockScaledLogisticRegression(feature_names=NAMES),
        lambda: logistic_regression_single_scaler(NAMES, scaler="standard"),
        lambda: logistic_regression_single_scaler(NAMES, scaler="maxabs"),
    ],
    ids=["block-scaled", "standard", "maxabs"],
)
def test_compiled_matches_pipelines(factory, data: tuple[np.ndarray, np.ndarray]) -> None:
    x, y = data
    classifier = MultilabelClassifier(factory).fit(x, y)

    compiled = LinearMultilabel.from_classifier(classifier)

    fresh = np.random.default_rng(1).normal(100, 80, (50, len(NAMES)))
    np.testing.assert_allclose(
        compiled.predict_proba(fresh), classifier.predict_proba(fresh), atol=1e-9
    )


def test_constant_label_keeps_its_constant_score(data: tuple[np.ndarray, np.ndarray]) -> None:
    x, y = data
    compiled = LinearMultilabel.from_classifier(
        MultilabelClassifier(BlockScaledLogisticRegression(feature_names=NAMES)).fit(x, y)
    )

    assert compiled.constant.tolist() == [False, False, False, True]
    assert np.all(compiled.predict_proba(x)[:, 3] == 0.0)


def test_shuffled_blocks_still_line_up() -> None:
    """ColumnTransformer emits blocks in its own order, not the input's. With
    descriptors before fingerprints in the input, a fold that ignored the reordering
    would multiply each weight by the wrong column."""
    rng = np.random.default_rng(2)
    names = ("MolWt", "morgan_bit_0", "TPSA", "morgan_bit_1")
    x = np.column_stack(
        [
            rng.normal(150, 50, 150),
            rng.integers(0, 2, 150),
            rng.normal(40, 10, 150),
            rng.integers(0, 2, 150),
        ]
    )
    y = np.column_stack([(x[:, 0] > 150), (x[:, 3] > 0)]).astype(np.int8)
    classifier = MultilabelClassifier(BlockScaledLogisticRegression(feature_names=names)).fit(x, y)

    compiled = LinearMultilabel.from_classifier(classifier)

    np.testing.assert_allclose(compiled.predict_proba(x), classifier.predict_proba(x), atol=1e-9)


def test_trees_are_refused_rather_than_approximated(data: tuple[np.ndarray, np.ndarray]) -> None:
    x, y = data
    classifier = MultilabelClassifier(lambda: gradient_boosting(max_iter=5)).fit(x, y)

    with pytest.raises(NotCompilable):
        LinearMultilabel.from_classifier(classifier)


def test_compiled_artifact_gives_the_same_probabilities(artifact: Path) -> None:
    from sillage.features import parse_smiles
    from sillage.features import sets as feature_sets

    model = OdorModel.load(artifact)
    x = feature_sets.build("descriptors", parse_smiles(["CCCCO", "c1ccccc1O", "CCSCC"])).values

    compiled = LinearMultilabel.from_classifier(model.estimator)

    np.testing.assert_allclose(
        model.calibrator.transform(compiled.predict_proba(x)),
        model.probabilities(x),
        atol=1e-5,
    )


# --- boosted trees (journal E-020) ----------------------------------------------------


def test_compiled_boosting_matches_scikit_learn(data: tuple[np.ndarray, np.ndarray]) -> None:
    from sillage.models.compiled import BoostedMultilabel
    from sillage.research import train_v2

    x, y = data
    y = y.astype(float)
    y[::7, 0] = np.nan  # not asked
    y[1::9, 1] = 0.5  # experts split
    matrix = type("Matrix", (), {"values": x})()
    boosting = train_v2.fit_boosting(matrix, y, np.arange(len(x)))
    probe = x.copy()
    probe[0, 0] = np.nan  # missing values take the side training sent them

    compiled = BoostedMultilabel.from_classifier(boosting)

    np.testing.assert_allclose(
        compiled.predict_proba(probe), boosting.predict_proba(probe), atol=1e-12
    )
    np.testing.assert_allclose(
        compiled.predict_proba(probe[:1]), boosting.predict_proba(probe[:1]), atol=1e-12
    )


def test_boosting_compiler_refuses_linear_models(data: tuple[np.ndarray, np.ndarray]) -> None:
    from sillage.models.compiled import BoostedMultilabel

    x, y = data
    linear = MultilabelClassifier(BlockScaledLogisticRegression(feature_names=NAMES)).fit(x, y)

    with pytest.raises(NotCompilable):
        BoostedMultilabel.from_classifier(linear)
