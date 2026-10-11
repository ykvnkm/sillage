"""Ports implemented over a saved :class:`OdorModel` artifact.

Nothing here is specific to logistic regression: the artifact records its own
feature set, calibrator and scope, and these adapters only replay them. A GNN
artifact will need its own predictor -- graphs instead of a feature matrix -- but
will reuse the scope policy unchanged.
"""

from __future__ import annotations

import numpy as np

from sillage.adapters.chem.reader import to_rdkit
from sillage.domain import DescriptorScore, ModelDescription, Molecule, OdorProfile, Refusal
from sillage.features import sets as feature_sets
from sillage.models.applicability import ChemicalScope
from sillage.models.artifact import ArtifactError, OdorModel
from sillage.models.compiled import LinearMultilabel, NotCompilable


class ArtifactOdorPredictor:
    """``OdorPredictor`` backed by a fingerprint-and-descriptor model artifact."""

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

        # 0.06 ms instead of 124 ms per molecule, identical answers (journal E-013).
        # A model that is not a stack of linear pipelines -- boosting -- falls back
        # to calling its estimators one by one.
        try:
            self._scorer = LinearMultilabel.from_classifier(model.estimator)
        except NotCompilable:
            self._scorer = None

    @property
    def name(self) -> str:
        return self._model.card.name

    @property
    def compiled(self) -> bool:
        return self._scorer is not None

    def predict(self, molecule: Molecule) -> OdorProfile:
        features = feature_sets.build(self._model.card.feature_set, [to_rdkit(molecule)])
        scores = self._probabilities(features.values)[0]
        return OdorProfile.from_scores(
            DescriptorScore(
                name=name,
                score=float(scores[index]),
                calibrated=bool(self._calibrated[index]),
                base_rate=float(self._base_rates[index]),
            )
            for index, name in enumerate(self._model.label_names)
        )

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

    def _probabilities(self, features: np.ndarray) -> np.ndarray:
        if self._scorer is None:
            return self._model.probabilities(features)
        if features.shape[1] != self._scorer.n_features_in_:
            raise ArtifactError(
                f"Model {self.name!r} expects {self._scorer.n_features_in_} features, "
                f"got {features.shape[1]}"
            )
        return self._calibrator.transform(self._scorer.predict_proba(features))


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
