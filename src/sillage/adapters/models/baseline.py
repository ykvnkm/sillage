"""Ports implemented over a saved :class:`OdorModel` artifact.

Nothing here is specific to logistic regression: the artifact records its own
feature set, calibrator and scope, and these adapters only replay them. A GNN
artifact will need its own predictor -- graphs instead of a feature matrix -- but
will reuse the scope policy unchanged.
"""

from __future__ import annotations

from sillage.adapters.chem.reader import to_rdkit
from sillage.domain import DescriptorScore, Molecule, OdorProfile, Refusal
from sillage.features import sets as feature_sets
from sillage.models.applicability import ChemicalScope
from sillage.models.artifact import ArtifactError, OdorModel


class ArtifactOdorPredictor:
    """``OdorPredictor`` backed by a fingerprint-and-descriptor model artifact."""

    def __init__(self, model: OdorModel) -> None:
        if model.calibrator is None:
            raise ArtifactError(
                f"Model {model.card.name!r} has no calibrator and cannot be served: "
                "its scores are not probabilities"
            )
        self._model = model
        self._calibrated = model.is_calibrated()
        self._base_rates = model.calibrator.base_rates

    @property
    def name(self) -> str:
        return self._model.card.name

    def predict(self, molecule: Molecule) -> OdorProfile:
        features = feature_sets.build(self._model.card.feature_set, [to_rdkit(molecule)])
        scores = self._model.probabilities(features.values)[0]
        return OdorProfile.from_scores(
            DescriptorScore(
                name=name,
                score=float(scores[index]),
                calibrated=bool(self._calibrated[index]),
                base_rate=float(self._base_rates[index]),
            )
            for index, name in enumerate(self._model.label_names)
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
