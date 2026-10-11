"""A saved model must reload into exactly the same model, or refuse to load at all.

Every failure mode here is silent by nature. A model that loads with the wrong
metadata, or returns slightly different numbers after a round trip, produces
plausible output and no error -- and the service ships predictions that do not match
the ones that were measured.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sillage.models import BlockScaledLogisticRegression, ChemicalScope, MultilabelClassifier
from sillage.models.artifact import (
    ARTIFACT_VERSION,
    CARD_FILE,
    ArtifactError,
    ModelCard,
    OdorModel,
    build_card,
)

FEATURE_NAMES = ("morgan_bit_0", "morgan_bit_1", "morgan_bit_2", "MolWt", "TPSA")
LABELS = ("fruity", "woody", "never")


@pytest.fixture
def trained_model() -> OdorModel:
    """A miniature of the shipping artefact: same classes, three features, three labels."""
    rng = np.random.default_rng(0)
    x = rng.random((80, len(FEATURE_NAMES)))
    y = np.zeros((80, len(LABELS)), dtype=np.int8)
    y[:, 0] = (x[:, 0] > 0.5).astype(np.int8)
    y[:, 1] = (x[:, 3] > 0.5).astype(np.int8)

    estimator = MultilabelClassifier(BlockScaledLogisticRegression(feature_names=FEATURE_NAMES))
    estimator.fit(x, y)

    card = build_card(
        name="test-model",
        feature_set="morgan-bin-r2+desc",
        estimator="logistic-regression-ovr-blockwise-scaled",
        seed=0,
        label_names=LABELS,
        data_source="fixture",
        data_sha256="0" * 64,
        split_strategy="stratified",
        n_train=80,
        n_test=20,
        metrics={"macro_average_precision": 0.5},
        baseline_metrics={"macro_average_precision": 0.1},
        scope=ChemicalScope(max_weight=350.0),
    )
    return OdorModel(
        card=card,
        estimator=estimator,
        scope=ChemicalScope(max_weight=350.0),
        reference_smiles=("CCO", "c1ccccc1"),
    )


def features(n: int = 12) -> np.ndarray:
    return np.random.default_rng(1).random((n, len(FEATURE_NAMES)))


# --- the round trip ----------------------------------------------------------


def test_predictions_survive_a_round_trip_exactly(trained_model: OdorModel, tmp_path: Path) -> None:
    """Not "close": identical. A deployed model that drifts from the measured one
    makes every number in the model card a lie."""
    x = features()
    before = trained_model.raw_scores(x)

    trained_model.save(tmp_path / "m")
    after = OdorModel.load(tmp_path / "m").raw_scores(x)

    np.testing.assert_array_equal(before, after)


def test_the_card_survives_a_round_trip(trained_model: OdorModel, tmp_path: Path) -> None:
    trained_model.save(tmp_path / "m")

    reloaded = OdorModel.load(tmp_path / "m")

    assert reloaded.card == trained_model.card


def test_the_scope_is_reconstructed(trained_model: OdorModel, tmp_path: Path) -> None:
    """The scope is policy, and policy has to travel with the weights."""
    trained_model.save(tmp_path / "m")

    assert OdorModel.load(tmp_path / "m").scope == ChemicalScope(max_weight=350.0)


def test_reference_molecules_are_kept(trained_model: OdorModel, tmp_path: Path) -> None:
    trained_model.save(tmp_path / "m")

    reloaded = OdorModel.load(tmp_path / "m")

    assert reloaded.reference_smiles == ("CCO", "c1ccccc1")
    assert len(reloaded.reference_molecules()) == 2


def test_saving_twice_overwrites_cleanly(trained_model: OdorModel, tmp_path: Path) -> None:
    trained_model.save(tmp_path / "m")
    trained_model.save(tmp_path / "m")

    assert OdorModel.load(tmp_path / "m").card.name == "test-model"


# --- refusing to load the wrong thing ----------------------------------------


def test_a_directory_that_is_not_an_artifact_is_refused(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()

    with pytest.raises(ArtifactError, match="not a model artifact"):
        OdorModel.load(tmp_path / "empty")


def test_a_missing_directory_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match="not a model artifact"):
        OdorModel.load(tmp_path / "absent")


def test_a_future_artifact_version_is_refused(trained_model: OdorModel, tmp_path: Path) -> None:
    """Reading an unfamiliar layout would mean guessing. Guessing is the failure."""
    trained_model.save(tmp_path / "m")
    card_path = tmp_path / "m" / CARD_FILE
    payload = json.loads(card_path.read_text())
    payload["artifact_version"] = ARTIFACT_VERSION + 1
    card_path.write_text(json.dumps(payload))

    with pytest.raises(ArtifactError, match="artifact version"):
        OdorModel.load(tmp_path / "m")


def test_the_wrong_number_of_features_is_refused(trained_model: OdorModel) -> None:
    """The error the service would otherwise never see: wrong features, happy numbers."""
    with pytest.raises(ArtifactError, match="expects 5 features"):
        trained_model.raw_scores(np.zeros((3, 7)))


# --- the card itself ---------------------------------------------------------


def test_the_card_records_the_data_it_was_trained_on(trained_model: OdorModel) -> None:
    """Two models with different data checksums are not comparable, whatever their
    metrics say."""
    assert trained_model.card.data_sha256 == "0" * 64
    assert trained_model.card.data_source == "fixture"


def test_the_card_records_the_floor_next_to_the_metric(trained_model: OdorModel) -> None:
    assert trained_model.card.baseline_metrics["macro_average_precision"] == 0.1


def test_the_card_is_json_round_trippable(trained_model: OdorModel) -> None:
    payload = json.loads(json.dumps(trained_model.card.to_dict()))

    assert ModelCard.from_dict(payload) == trained_model.card


def test_label_names_are_exposed_in_order(trained_model: OdorModel) -> None:
    """Column order is the only thing that connects a score to a descriptor."""
    assert trained_model.label_names == LABELS


# --- the reason the factory is a class ---------------------------------------


def test_the_estimator_factory_is_picklable() -> None:
    """A lambda here raised PicklingError and made the model unshippable."""
    import pickle

    factory = BlockScaledLogisticRegression(feature_names=FEATURE_NAMES, seed=3)

    assert pickle.loads(pickle.dumps(factory)) == factory


def test_the_factory_keeps_its_configuration() -> None:
    factory = BlockScaledLogisticRegression(feature_names=FEATURE_NAMES, c=0.5, seed=7)

    pipeline = factory()

    assert pipeline.named_steps["model"].C == 0.5
    assert pipeline.named_steps["model"].random_state == 7
