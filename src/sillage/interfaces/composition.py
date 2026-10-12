"""The composition root: the one place that knows which adapter fills which port.

Every entry point -- the CLI, the FastAPI app -- builds what it needs here and
nowhere else. Swapping the baseline for the GNN is a change to this file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from sillage.adapters.chem import (
    FingerprintReferenceLibrary,
    RdkitDepicter,
    RdkitMoleculeReader,
)
from sillage.adapters.chem.names import NAMES_FILE, NameIndex
from sillage.adapters.models import ArtifactOdorPredictor, ArtifactScopePolicy
from sillage.adapters.models.odor_map import MAP_FILE, EmbeddingMap
from sillage.adapters.pubchem import PubChemResolver
from sillage.application import PredictOdor
from sillage.domain.ports import MoleculeDepicter, NameDirectory, OdorMap
from sillage.models.artifact import OdorModel
from sillage.paths import model_dir

ONLINE_NAMES_ENV = "SILLAGE_ONLINE_NAMES"
"""Set to ``0`` to answer only from the local name directory, never calling PubChem."""


@dataclass(frozen=True, slots=True)
class Services:
    """Everything the HTTP app serves, built once at startup."""

    predict: PredictOdor
    depict: MoleculeDepicter
    names: NameDirectory | None
    odor_map: OdorMap | None


def build_predict_odor(directory: Path | None = None, *, online: bool | None = None) -> PredictOdor:
    """Load an artifact and wire the prediction use case around it.

    Takes about a second, most of it fingerprinting the reference library. Call it
    once at startup, not per request.

    *online* enables PubChem for names the local directory lacks; by default it
    follows ``$SILLAGE_ONLINE_NAMES`` and is on.
    """
    directory = directory if directory is not None else model_dir()
    model = OdorModel.load(directory)
    if online is None:
        online = os.environ.get(ONLINE_NAMES_ENV, "1") != "0"
    names_file = directory / NAMES_FILE
    map_file = directory / MAP_FILE
    return PredictOdor(
        reader=RdkitMoleculeReader(),
        scope=ArtifactScopePolicy(model.scope),
        predictor=ArtifactOdorPredictor(model),
        references=FingerprintReferenceLibrary(model.reference),
        # Optional: a model published before step 2.7 has no names.
        names=NameIndex.from_file(names_file) if names_file.exists() else None,
        resolver=PubChemResolver() if online else None,
        # Optional too: built by scripts/build_map.py, absent before step 2.8.
        odor_map=EmbeddingMap.from_file(map_file) if map_file.exists() else None,
    )


def build_services(directory: Path | None = None, *, online: bool | None = None) -> Services:
    predict = build_predict_odor(directory, online=online)
    return Services(
        predict=predict, depict=RdkitDepicter(), names=predict.names, odor_map=predict.odor_map
    )
