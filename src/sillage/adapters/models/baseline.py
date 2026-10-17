"""Ports implemented over a saved :class:`OdorModel` artifact.

Nothing here is specific to one model family: the artifact records its feature
sets, components, calibrator and scope, and these adapters only replay them. A model
with graph networks reads the molecule with its stereochemistry; one without cannot,
and says so (:attr:`ArtifactOdorPredictor.sees_stereo`).
"""

from __future__ import annotations

import numpy as np

from sillage.adapters.chem.reader import to_rdkit
from sillage.domain import DescriptorScore, ModelDescription, Molecule, OdorProfile, Refusal
from sillage.models.applicability import ChemicalScope
from sillage.models.artifact import ArtifactError, OdorModel


class ArtifactOdorPredictor:
    """``OdorPredictor`` backed by a saved model artifact, whatever its components."""

    def __init__(self, model: OdorModel) -> None:
        if model.calibrator is None:
            raise ArtifactError(
                f"Model {model.card.name!r} has no calibrator and cannot be served: "
                "its scores are not probabilities"
            )
        self._model = model
        self._calibrator = model.calibrator
        self._calibrated = model.is_calibrated()
        self._base_rates = model.calibrator.base_rates

        self._scorer = model.scorer()

    @property
    def name(self) -> str:
        return self._model.card.name

    @property
    def compiled(self) -> bool:
        return not hasattr(self._scorer.linear, "estimators_")

    @property
    def sees_stereo(self) -> bool:
        return self._scorer.sees_stereo

    def predict(self, molecule: Molecule) -> OdorProfile:
        scores = self._calibrator.transform(self._scorer.raw([to_rdkit(molecule)]))[0]
        return OdorProfile.from_scores(
            DescriptorScore(
                name=name,
                score=float(scores[index]),
                calibrated=bool(self._calibrated[index]),
                base_rate=float(self._base_rates[index]),
            )
            for index, name in enumerate(self._model.label_names)
        )

    def embed(self, molecule: Molecule) -> tuple[float, ...]:
        """Raw predicted probabilities, before calibration: the model's own scale,
        and exactly what scripts/build_map.py embedded the reference molecules with."""
        return tuple(float(v) for v in self._scorer.raw([to_rdkit(molecule)])[0])

    def describe(self) -> ModelDescription:
        card = self._model.card
        labels = np.asarray(card.label_names)
        return ModelDescription(
            name=card.name,
            descriptors=card.label_names,
            unverified=tuple(labels[~self._calibrated]),
            scope=describe_scope(self._model.scope),
            trained_on=card.n_train,
            metrics=dict(card.metrics),
            floor=dict(card.baseline_metrics),
        )


class ArtifactScopePolicy:
    """``ScopePolicy`` that applies the scope saved alongside the model's weights."""

    def __init__(self, scope: ChemicalScope) -> None:
        self._scope = scope

    def check(self, molecule: Molecule) -> Refusal | None:
        found = self._scope.violation(to_rdkit(molecule))
        if found is None:
            return None
        reason, detail = found
        return Refusal(reason=reason, detail=detail, molecule=molecule)


def describe_scope(scope: ChemicalScope) -> str:
    parts = []
    if scope.require_single_component:
        parts.append("single-component")
    if scope.require_neutral:
        parts.append("neutral")
    if scope.require_carbon:
        parts.append("organic")
    sentence = " ".join([*parts, "molecules"])
    sentence += f", {scope.min_weight:.0f}-{scope.max_weight:.0f} Da"
    if scope.allowed_elements:
        sentence += f", made only of {', '.join(scope.allowed_elements)}"
    return sentence
