"""A trained model that lives in a file instead of inside a script.

The point of this module is that a saved model is **self-describing**. Loading one
gives you not only the weights but everything needed to use them correctly: which
feature set it expects, what its 138 output columns mean, which revision of which
dataset it saw, what it scored, and the chemical scope it is defined on.

Without those, a loaded model is dangerous rather than useful -- feed it a different
feature set and it returns confident numbers about nothing, with no error anywhere.

Layout on disk::

    models/<name>/
        card.json        metadata, metrics, label names, data provenance
        estimator.joblib the fitted one-vs-rest classifier
        reference.json   every in-scope dataset molecule with its expert labels

Weights are not committed: `/models/` is excluded in `.gitignore`, and the card is
reproduced as a human-readable model card under `docs/`.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from rdkit.Chem import Mol

from sillage.features import parse_smiles
from sillage.models.applicability import ChemicalScope
from sillage.models.calibration import PerLabelCalibrator
from sillage.models.multilabel import MultilabelClassifier

ARTIFACT_VERSION = 3
"""Bumped whenever the on-disk layout changes. A mismatch refuses to load rather
than silently misreading an old directory."""

CARD_FILE = "card.json"
CALIBRATOR_FILE = "calibrator.joblib"
ESTIMATOR_FILE = "estimator.joblib"
REFERENCE_FILE = "reference.json"


class ArtifactError(RuntimeError):
    """Raised when a saved model cannot be loaded safely."""


@dataclass(frozen=True, slots=True)
class ReferenceMolecule:
    """A dataset molecule the product can point at: "the closest thing I know"."""

    smiles: str
    """Canonical, without stereochemistry -- the same form the service reads input into."""

    descriptors: tuple[str, ...]
    """Expert labels. What the dataset says, not what the model predicts."""

    split: str
    """``train``, ``val`` or ``test``. Every part is shown to users -- a neighbour is a
    fact about the dataset, and the measurements are already done -- but monitoring
    asks "how far is this input from what the model *learnt* on", and for that only
    ``train`` counts."""


@dataclass(frozen=True, slots=True)
class ModelCard:
    """Everything a caller needs to interpret this model's output."""

    name: str
    artifact_version: int
    created_at: str

    feature_set: str
    """Name in :data:`sillage.features.sets.FEATURE_SETS`. The model is only
    valid on features built by that exact entry."""

    estimator: str
    seed: int
    label_names: tuple[str, ...]

    data_source: str
    data_sha256: str
    """Pinned checksum of the training data. Two models with different values here
    are not comparable, whatever their metrics say."""

    split_strategy: str
    n_train: int
    n_test: int

    metrics: dict[str, float]
    baseline_metrics: dict[str, float]
    """What a predictor that never saw a molecule scores on the same split. A metric
    without its floor is unreadable -- micro AUC starts at 0.78 here (journal E-004)."""

    scope: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["label_names"] = list(self.label_names)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ModelCard:
        data = dict(payload)
        data["label_names"] = tuple(data["label_names"])
        return cls(**data)


@dataclass(frozen=True, slots=True)
class OdorModel:
    """A fitted model plus the context that makes it safe to call."""

    card: ModelCard
    estimator: MultilabelClassifier
    scope: ChemicalScope
    calibrator: PerLabelCalibrator | None
    """Fitted on the validation split. Without it the raw scores are not
    probabilities: uncalibrated, a 0.94 is right 83% of the time (journal E-011)."""

    reference: tuple[ReferenceMolecule, ...]
    """Dataset molecules, kept so the product can answer "the closest thing I know is
    X, and experts call it Y". Fingerprints are recomputed on load rather than
    stored: it takes under a second and removes a second thing that can fall out of
    sync."""

    @property
    def label_names(self) -> tuple[str, ...]:
        return self.card.label_names

    def raw_scores(self, features: np.ndarray) -> np.ndarray:
        """Uncalibrated per-label scores for an already-built feature matrix.

        Deliberately low-level and deliberately named *raw*: these numbers are not
        probabilities. The product must never show them to a user unmediated.
        """
        if features.shape[1] != self._expected_width():
            raise ArtifactError(
                f"Model {self.card.name!r} expects {self._expected_width()} features "
                f"(feature set {self.card.feature_set!r}), got {features.shape[1]}"
            )
        return self.estimator.predict_proba(features)

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        """Calibrated probabilities. The only output the product may show."""
        scores = self.raw_scores(features)
        if self.calibrator is None:
            raise ArtifactError(
                f"Model {self.card.name!r} has no calibrator; raw scores must not be "
                "presented as probabilities"
            )
        return self.calibrator.transform(scores)

    def is_calibrated(self) -> np.ndarray:
        """Per-label flag: True where the probability may be stated as such."""
        if self.calibrator is None:
            return np.zeros(len(self.label_names), dtype=bool)
        return self.calibrator.fitted

    def _expected_width(self) -> int:
        for fitted in self.estimator.estimators_:
            if fitted is not None:
                return int(fitted.n_features_in_)
        raise ArtifactError("Model has no fitted estimators")

    @property
    def reference_smiles(self) -> tuple[str, ...]:
        return tuple(entry.smiles for entry in self.reference)

    def reference_molecules(self) -> list[Mol]:
        return parse_smiles(self.reference_smiles)

    # --- persistence ---------------------------------------------------------

    def save(self, directory: Path) -> Path:
        """Write the artifact. Overwrites an existing directory of the same name."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        (directory / CARD_FILE).write_text(
            json.dumps(self.card.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (directory / REFERENCE_FILE).write_text(
            json.dumps([asdict(entry) for entry in self.reference], ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        joblib.dump(self.estimator, directory / ESTIMATOR_FILE, compress=3)
        joblib.dump(self.calibrator, directory / CALIBRATOR_FILE, compress=3)
        return directory

    @classmethod
    def load(cls, directory: Path) -> OdorModel:
        """Read an artifact, refusing anything it cannot interpret.

        Every failure here is loud on purpose. A model loaded from a directory
        written by a different version of this code would produce plausible numbers
        and no warning, which is the worst possible outcome.
        """
        directory = Path(directory)
        missing = [
            name
            for name in (CARD_FILE, ESTIMATOR_FILE, REFERENCE_FILE, CALIBRATOR_FILE)
            if not (directory / name).exists()
        ]
        if missing:
            raise ArtifactError(
                f"{directory} is not a model artifact: missing {', '.join(missing)}"
            )

        card = ModelCard.from_dict(json.loads((directory / CARD_FILE).read_text(encoding="utf-8")))
        if card.artifact_version != ARTIFACT_VERSION:
            raise ArtifactError(
                f"{directory} has artifact version {card.artifact_version}, "
                f"this code reads version {ARTIFACT_VERSION}. Retrain rather than guess."
            )

        return cls(
            card=card,
            estimator=joblib.load(directory / ESTIMATOR_FILE),
            calibrator=joblib.load(directory / CALIBRATOR_FILE),
            scope=_scope_from_card(card),
            reference=tuple(
                ReferenceMolecule(
                    smiles=entry["smiles"],
                    descriptors=tuple(entry["descriptors"]),
                    split=entry["split"],
                )
                for entry in json.loads((directory / REFERENCE_FILE).read_text(encoding="utf-8"))
            ),
        )


def _scope_from_card(card: ModelCard) -> ChemicalScope:
    """Rebuild the scope, restoring the tuple that JSON flattened into a list."""
    payload = dict(card.scope)
    elements = payload.get("allowed_elements")
    if elements is not None:
        payload["allowed_elements"] = tuple(elements)
    return ChemicalScope(**payload)


def build_card(
    *,
    name: str,
    feature_set: str,
    estimator: str,
    seed: int,
    label_names: Sequence[str],
    data_source: str,
    data_sha256: str,
    split_strategy: str,
    n_train: int,
    n_test: int,
    metrics: dict[str, float],
    baseline_metrics: dict[str, float],
    scope: ChemicalScope,
) -> ModelCard:
    """Assemble a card, stamping the version and the creation time."""
    return ModelCard(
        name=name,
        artifact_version=ARTIFACT_VERSION,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        feature_set=feature_set,
        estimator=estimator,
        seed=seed,
        label_names=tuple(label_names),
        data_source=data_source,
        data_sha256=data_sha256,
        split_strategy=split_strategy,
        n_train=n_train,
        n_test=n_test,
        metrics=metrics,
        baseline_metrics=baseline_metrics,
        scope=asdict(scope),
    )
