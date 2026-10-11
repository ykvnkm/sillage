"""The composition root: the one place that knows which adapter fills which port.

Every entry point -- this CLI, the FastAPI app of step 2.4 -- builds its use case
here and nowhere else. Swapping the baseline for the GNN is a change to this file.
"""

from __future__ import annotations

from pathlib import Path

from sillage.adapters.chem import FingerprintReferenceLibrary, RdkitMoleculeReader
from sillage.adapters.models import ArtifactOdorPredictor, ArtifactScopePolicy
from sillage.application import PredictOdor
from sillage.models.artifact import OdorModel
from sillage.paths import model_dir


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
