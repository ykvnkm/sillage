"""The composition root: the one place that knows which adapter fills which port.

Every entry point -- the CLI, the FastAPI app -- builds what it needs here and
nowhere else. Swapping the baseline for the GNN is a change to this file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sillage.adapters.chem import (
    FingerprintReferenceLibrary,
    RdkitDepicter,
    RdkitMoleculeReader,
)
from sillage.adapters.models import ArtifactOdorPredictor, ArtifactScopePolicy
from sillage.application import PredictOdor
from sillage.domain.ports import MoleculeDepicter
from sillage.models.artifact import OdorModel
from sillage.paths import model_dir


@dataclass(frozen=True, slots=True)
class Services:
    """Everything the HTTP app serves, built once at startup."""

    predict: PredictOdor
    depict: MoleculeDepicter


def build_predict_odor(directory: Path | None = None) -> PredictOdor:
    """Load an artifact and wire the prediction use case around it.

    Takes about a second, most of it fingerprinting the reference library. Call it
    once at startup, not per request.
    """
    model = OdorModel.load(directory if directory is not None else model_dir())
    return PredictOdor(
        reader=RdkitMoleculeReader(),
        scope=ArtifactScopePolicy(model.scope),
        predictor=ArtifactOdorPredictor(model),
        references=FingerprintReferenceLibrary(model.reference),
    )


def build_services(directory: Path | None = None) -> Services:
    return Services(predict=build_predict_odor(directory), depict=RdkitDepicter())
