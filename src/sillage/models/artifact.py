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
        estimator.joblib the fitted one-vs-rest linear classifier
        boosting.joblib  optional: boosting on descriptors (journal E-020)
        graph.npz        optional: graph networks, plain arrays (journal E-020)
        calibrator.joblib per-label calibration of the components' mean
        reference.json   every in-scope dataset molecule with its expert labels

When several components are present, the model's raw score is their plain mean;
the calibrator is fitted on that mean.

Weights are not committed: `/models/` is excluded in `.gitignore`, and the card is
reproduced as a human-readable model card under `docs/`.
"""

from __future__ import annotations

import contextlib
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
from sillage.features import sets as feature_sets
from sillage.features.graphs import molecular_graph
from sillage.models.applicability import ChemicalScope
from sillage.models.calibration import PerLabelCalibrator
from sillage.models.compiled import BoostedMultilabel, LinearMultilabel, NotCompilable
from sillage.models.graph_network import GraphEnsemble
from sillage.models.multilabel import MultilabelClassifier

ARTIFACT_VERSION = 4
"""Bumped whenever the on-disk layout changes. A mismatch refuses to load rather
than silently misreading an old directory."""

READABLE_VERSIONS = frozenset({3, ARTIFACT_VERSION})
"""Version 3 is version 4 with the linear component only, so it reads as exactly that.
The reverse is what the bump guards: code that knows only version 3 would open a
version-4 ensemble and silently serve its linear part."""

CARD_FILE = "card.json"
CALIBRATOR_FILE = "calibrator.joblib"
ESTIMATOR_FILE = "estimator.joblib"
REFERENCE_FILE = "reference.json"
BOOSTING_FILE = "boosting.joblib"
GRAPH_FILE = "graph.npz"


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

    components: tuple[str, ...] = ("linear",)
    """Which parts the raw score averages: ``linear``, ``boosting``, ``graph``."""

    boosting_feature_set: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["label_names"] = list(self.label_names)
        payload["components"] = list(self.components)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ModelCard:
        data = dict(payload)
        data["label_names"] = tuple(data["label_names"])
        data["components"] = tuple(data.get("components", ("linear",)))
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

    boosting: MultilabelClassifier | None = None
    graph: GraphEnsemble | None = None

    @property
    def label_names(self) -> tuple[str, ...]:
        return self.card.label_names

    def scorer(self) -> EnsembleScorer:
        """What the service calls: every component, from RDKit molecules to raw scores."""
        return EnsembleScorer.from_model(self)

    def raw_scores(self, features: np.ndarray) -> np.ndarray:
        """Uncalibrated per-label scores of the **linear component**, for an
        already-built feature matrix. A model with other components is scored with
        :meth:`scorer` instead.

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
        if self.boosting is not None:
            joblib.dump(self.boosting, directory / BOOSTING_FILE, compress=3)
        if self.graph is not None:
            self.graph.save(directory / GRAPH_FILE)
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
        if card.artifact_version not in READABLE_VERSIONS:
            raise ArtifactError(
                f"{directory} has artifact version {card.artifact_version}, this code "
                f"reads versions {sorted(READABLE_VERSIONS)}. Retrain rather than guess."
            )

        for component, file in (("boosting", BOOSTING_FILE), ("graph", GRAPH_FILE)):
            if component in card.components and not (directory / file).exists():
                raise ArtifactError(f"{directory}: the card lists {component}, {file} is missing")

        return cls(
            card=card,
            estimator=joblib.load(directory / ESTIMATOR_FILE),
            boosting=(
                joblib.load(directory / BOOSTING_FILE) if "boosting" in card.components else None
            ),
            graph=(
                GraphEnsemble.load(directory / GRAPH_FILE) if "graph" in card.components else None
            ),
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
    components: Sequence[str] = ("linear",),
    boosting_feature_set: str | None = None,
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
        components=tuple(components),
        boosting_feature_set=boosting_feature_set,
    )


@dataclass(frozen=True, slots=True)
class EnsembleScorer:
    """Raw scores for RDKit molecules: the mean of whichever components the model has.

    The linear part is compiled to one matrix product (journal E-013); boosting and
    the graph networks are called as they are.
    """

    feature_set: str
    linear: LinearMultilabel | MultilabelClassifier
    boosting: BoostedMultilabel | MultilabelClassifier | None = None
    boosting_feature_set: str | None = None
    graph: GraphEnsemble | None = None

    @classmethod
    def from_model(cls, model: OdorModel) -> EnsembleScorer:
        try:
            linear = LinearMultilabel.from_classifier(model.estimator)
        except NotCompilable:
            linear = model.estimator
        boosting = model.boosting
        if boosting is not None:
            # 60 ms -> a few ms per molecule, identical answers (journal E-020).
            with contextlib.suppress(NotCompilable):
                boosting = BoostedMultilabel.from_classifier(boosting)
        return cls(
            feature_set=model.card.feature_set,
            linear=linear,
            boosting=boosting,
            boosting_feature_set=model.card.boosting_feature_set,
            graph=model.graph,
        )

    @property
    def sees_stereo(self) -> bool:
        return self.graph is not None

    def raw(self, molecules: Sequence[Mol]) -> np.ndarray:
        parts = [self.linear.predict_proba(feature_sets.build(self.feature_set, molecules).values)]
        if self.boosting is not None:
            features = feature_sets.build(self.boosting_feature_set, molecules).values
            parts.append(self.boosting.predict_proba(features))
        if self.graph is not None:
            parts.append(np.stack([self.graph.predict(molecular_graph(m)) for m in molecules]))
        return np.mean(parts, axis=0)
