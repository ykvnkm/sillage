"""Shared fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sillage.features import parse_smiles
from sillage.features import sets as feature_sets
from sillage.models import (
    BlockScaledLogisticRegression,
    ChemicalScope,
    MultilabelClassifier,
    PerLabelCalibrator,
)
from sillage.models.artifact import OdorModel, ReferenceMolecule, build_card

SMILES = (
    # alcohols, aldehydes, acids
    *("CCO", "CCCO", "CCCCO", "CCCCCO", "CC(C)O", "CC=O", "CCC=O", "CCCC=O"),
    *("CC(=O)O", "CCC(=O)O"),
    # alkanes
    *("CCCCC", "CCCCCC", "CCCCCCC", "CC(C)C", "CC(C)CC"),
    # aromatics
    *("c1ccccc1", "Cc1ccccc1", "CCc1ccccc1", "Oc1ccccc1", "O=Cc1ccccc1", "OCc1ccccc1"),
    *("COc1ccccc1", "CC(=O)c1ccccc1", "Cc1ccc(C)cc1", "Nc1ccccc1", "c1ccncc1"),
    # sulfur and nitrogen
    *("CSC", "CCSCC", "CCN", "CCCN"),
)
LABELS = ("oxygenated", "aromatic", "sulfurous")

# A small name directory in the format scripts/build_names.py writes.
NAMES = {
    "CCCCO": {"cid": 263, "en": "1-Butanol", "ru": "бутанол", "aliases": ["n-Butyl alcohol"]},
    "c1ccccc1": {"cid": 241, "en": "Benzene", "ru": "бензол", "aliases": []},
    "CCSCC": {"cid": 9609, "en": "Diethyl sulfide", "ru": None, "aliases": ["Ethyl sulfide"]},
    "O=Cc1ccccc1": {"cid": 240, "en": "Benzaldehyde", "ru": None, "aliases": []},
}


def _labels(smiles: str) -> list[int]:
    return [int("O" in smiles or "o" in smiles), int("c" in smiles), int("S" in smiles)]


@pytest.fixture(scope="session")
def artifact(tmp_path_factory: pytest.TempPathFactory) -> Path:
    molecules = parse_smiles(SMILES)
    features = feature_sets.build("descriptors", molecules)
    y = np.array([_labels(s) for s in SMILES], dtype=np.int8)

    estimator = MultilabelClassifier(BlockScaledLogisticRegression(feature_names=features.names))
    estimator.fit(features.values, y)
    scope = ChemicalScope(allowed_elements=("C", "N", "O", "S"))
    card = build_card(
        name="mini",
        feature_set="descriptors",
        estimator="logistic-regression-ovr-blockwise-scaled",
        seed=0,
        label_names=LABELS,
        data_source="fixture",
        data_sha256="0" * 64,
        split_strategy="fixture",
        n_train=len(SMILES),
        n_test=0,
        metrics={},
        baseline_metrics={},
        scope=scope,
    )
    model = OdorModel(
        card=card,
        estimator=estimator,
        # Fitted on training data, which is wrong for a real model and fine for wiring.
        # `sulfurous` has 2 positives, below the 6 needed: it stays uncalibrated.
        calibrator=PerLabelCalibrator.fit(
            estimator.predict_proba(features.values), y, base_rates=y.mean(axis=0)
        ),
        scope=scope,
        reference=tuple(
            ReferenceMolecule(s, tuple(np.asarray(LABELS)[row == 1]), "train" if i % 5 else "test")
            for i, (s, row) in enumerate(zip(SMILES, y, strict=True))
        ),
    )
    directory = model.save(tmp_path_factory.mktemp("artifact") / "mini")
    (directory / "names.json").write_text(json.dumps(NAMES, ensure_ascii=False), "utf-8")
    write_map(directory, model, features.values)
    return directory


def write_map(directory: Path, model: OdorModel, features: np.ndarray) -> None:
    """A map in the format scripts/build_map.py writes. Positions are arbitrary here --
    UMAP's layout is not under test -- but embeddings are the model's real ones, so
    placing a new molecule exercises the same arithmetic as the service."""
    from sillage.domain.families import Family
    from sillage.models.compiled import LinearMultilabel

    embeddings = LinearMultilabel.from_classifier(model.estimator).predict_proba(features)
    rng = np.random.default_rng(0)
    np.savez_compressed(
        directory / "map.npz",
        smiles=np.array([r.smiles for r in model.reference]),
        coords=rng.random((len(model.reference), 2)).astype(np.float32),
        embeddings=embeddings.astype(np.float16),
        families=np.array([i % 3 for i in range(len(model.reference))], dtype=np.int8),
        family_order=np.array([f.value for f in Family]),
    )


@pytest.fixture(autouse=True)
def offline_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test may depend on PubChem being up: online name lookups are off everywhere.
    Tests of the PubChem adapter stub the network themselves."""
    monkeypatch.setenv("SILLAGE_ONLINE_NAMES", "0")
