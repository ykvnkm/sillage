"""Shared fixtures."""

from __future__ import annotations

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
    return model.save(tmp_path_factory.mktemp("artifact") / "mini")
