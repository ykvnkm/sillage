"""Training the artefact the product ships.

Research code trains; the product only loads. That split is why this lives here and
:mod:`sillage.models.artifact` lives there -- the service must never import a
training path, and on CPU inference it must not pull the training stack at all.

The configuration below is not a guess. It is the winner of the phase 1 ablation:
block-wise scaled logistic regression on `morgan-bin-r3+desc`, measured at
macro AP 0.3420 +/- 0.0126 against a floor of 0.0349 (journal E-005, E-006).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from sillage.data.sources import OPENPOM_CURATED
from sillage.features import sets as feature_sets
from sillage.metrics import MultilabelReport, evaluate
from sillage.models import (
    BlockScaledLogisticRegression,
    ChemicalScope,
    MultilabelClassifier,
    PerLabelCalibrator,
    PrevalenceBaseline,
    element_vocabulary,
)
from sillage.models.artifact import OdorModel, ReferenceMolecule, build_card
from sillage.research.baseline import Dataset
from sillage.splits import Split

DEFAULT_NAME = "odor-baseline-v1"
DEFAULT_FEATURE_SET = "morgan-bin-r3+desc"
DEFAULT_SPLIT = "stratified"
DEFAULT_SEED = 0


@dataclass(frozen=True, slots=True)
class TrainingOutcome:
    """The artefact plus the reports that justify shipping it."""

    model: OdorModel
    report: MultilabelReport
    baseline_report: MultilabelReport

    def summary(self) -> str:
        lines = [
            f"model:    {self.model.card.name}  ({self.model.card.estimator})",
            f"features: {self.model.card.feature_set}",
            f"trained:  {self.model.card.n_train} molecules, tested on {self.model.card.n_test}",
            f"calibrated: {self.model.calibrator.n_fitted} of "
            f"{self.model.calibrator.n_labels} labels"
            if self.model.calibrator is not None
            else "calibrated: no",
            "",
            "                    model     floor",
            f"macro AP          {self.report.macro_average_precision:>7.4f}   "
            f"{self.baseline_report.macro_average_precision:>7.4f}",
            f"macro AUC         {self.report.macro_roc_auc:>7.4f}   "
            f"{self.baseline_report.macro_roc_auc:>7.4f}",
            f"micro AP          {self.report.micro_average_precision:>7.4f}   "
            f"{self.baseline_report.micro_average_precision:>7.4f}",
            f"micro AUC         {self.report.micro_roc_auc:>7.4f}   "
            f"{self.baseline_report.micro_roc_auc:>7.4f}",
            "",
            f"labels scored: {self.report.n_labels_scored} of {self.report.n_labels_total}",
        ]
        return "\n".join(lines)


def train_odor_model(
    *,
    name: str = DEFAULT_NAME,
    feature_set: str = DEFAULT_FEATURE_SET,
    split_strategy: str = DEFAULT_SPLIT,
    seed: int = DEFAULT_SEED,
    scope: ChemicalScope | None = None,
    restrict_to_scope: bool = True,
    data: Dataset | None = None,
) -> TrainingOutcome:
    """Fit the shipping model and measure it against the floor on the same split.

    Trained on the training split only. The validation split is left untouched on
    purpose: step 2.2 needs held-out data to calibrate probabilities on, and
    calibrating on data the model was fitted to would produce confident nonsense
    that looks well calibrated.
    """
    data = data if data is not None else Dataset.load()
    scope = scope if scope is not None else ChemicalScope()

    # Train on the distribution the product actually serves. Fitting on molecules
    # the service refuses to answer about spends capacity on inputs it will never
    # see, and reports metrics over molecules no user can ask for (journal E-009).
    #
    # Two passes, because the element vocabulary is a property of the training data
    # rather than of chemistry: first the rules that stand on their own, then the
    # vocabulary derived from what survived, then the vocabulary rule itself.
    if restrict_to_scope:
        data = data.restrict(data.in_scope(scope))
        scope = replace(scope, allowed_elements=element_vocabulary(data.molecules))
        data = data.restrict(data.in_scope(scope))

    features = feature_sets.build(feature_set, data.molecules)
    split = data.split(split_strategy, seed)

    x_train, y_train = features.values[split.train], data.labels[split.train]
    x_test, y_test = features.values[split.test], data.labels[split.test]

    factory = BlockScaledLogisticRegression(feature_names=features.names, seed=seed)
    estimator = MultilabelClassifier(factory)
    estimator.fit(x_train, y_train)
    report = evaluate(y_test, estimator.predict_proba(x_test), data.label_names)

    # Calibrated on the validation split, which training never touched. Fitting the
    # calibrator on training data yields a model that looks perfectly calibrated and
    # is not: it has memorised those answers (journal E-011).
    calibrator = PerLabelCalibrator.fit(
        estimator.predict_proba(features.values[split.val]),
        data.labels[split.val],
        base_rates=y_train.mean(axis=0),
    )

    floor = PrevalenceBaseline().fit(x_train, y_train)
    baseline_report = evaluate(y_test, floor.predict_proba(x_test), data.label_names)

    card = build_card(
        name=name,
        feature_set=feature_set,
        estimator="logistic-regression-ovr-blockwise-scaled",
        seed=seed,
        label_names=data.label_names,
        data_source=OPENPOM_CURATED.name,
        data_sha256=OPENPOM_CURATED.sha256 or "unpinned",
        split_strategy=split_strategy + ("+in-scope" if restrict_to_scope else ""),
        n_train=int(split.train.size),
        n_test=int(split.test.size),
        metrics=_metrics(report),
        baseline_metrics=_metrics(baseline_report),
        scope=scope,
    )

    model = OdorModel(
        card=card,
        estimator=estimator,
        calibrator=calibrator,
        scope=scope,
        reference=_reference_library(data, split),
    )
    return TrainingOutcome(model=model, report=report, baseline_report=baseline_report)


def _metrics(report: MultilabelReport) -> dict[str, float]:
    return {
        "macro_average_precision": float(report.macro_average_precision),
        "macro_roc_auc": float(report.macro_roc_auc),
        "micro_average_precision": float(report.micro_average_precision),
        "micro_roc_auc": float(report.micro_roc_auc),
        "labels_scored": float(report.n_labels_scored),
        "labels_total": float(report.n_labels_total),
    }


def _reference_library(data: Dataset, split: Split) -> tuple[ReferenceMolecule, ...]:
    """Every in-scope molecule with its expert labels and the part it landed in.

    SMILES are written without stereochemistry, the form the service reads input
    into, so that an exact-match lookup compares like with like.
    """
    from rdkit import Chem

    part_of = np.empty(len(data.molecules), dtype=object)
    for part, indices in split.parts().items():
        part_of[indices] = part

    names = np.asarray(data.label_names)
    return tuple(
        ReferenceMolecule(
            smiles=Chem.MolToSmiles(molecule, isomericSmiles=False),
            descriptors=tuple(names[data.labels[index] == 1]),
            split=str(part_of[index]),
        )
        for index, molecule in enumerate(data.molecules)
    )
