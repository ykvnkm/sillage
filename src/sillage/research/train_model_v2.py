"""Training the shipping model on dataset v2 (journal E-019).

Same estimator family as odor-baseline-v1 -- block-scaled logistic regression on
Morgan bits and descriptors -- trained on the two expert collections rebuilt from
their own records: a label a collection could not express is missing, not "no", and
a label the experts split on is a soft 0.5. Five paired splits put this ahead of v1's
recipe on every one: +0.011 macro AP on experts' consensus, +0.012 against Leffingwell.

What ships alongside is unchanged in kind: calibration (fitted on the validation part,
unknown cells skipped), the declared scope, and a reference library. The library keeps
OpenPOM's records -- MIT-licensed, publishable -- while the Pyrfume collections the
model learns from are derived-only (see sources.py): their records never leave the
machine, only the weights do.
"""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass

import numpy as np
from rdkit import Chem

from sillage.data.sources import PYRFUME_SOURCES
from sillage.features import sets as feature_sets
from sillage.models import PerLabelCalibrator
from sillage.models.artifact import OdorModel, ReferenceMolecule, build_card
from sillage.research import train_v2
from sillage.research.baseline import Dataset
from sillage.research.dataset_v2 import DatasetV2, build

DEFAULT_NAME = "odor-baseline-v2"
DEFAULT_FEATURE_SET = "morgan-bin-r3+desc"
DEFAULT_SEED = 0
DEFAULT_C = 0.1


@dataclass(frozen=True, slots=True)
class TrainingOutcomeV2:
    model: OdorModel
    metrics: dict[str, float]
    floor: dict[str, float]

    def summary(self) -> str:
        card = self.model.card
        calibrator = self.model.calibrator
        lines = [
            f"model:    {card.name}  ({card.estimator})",
            f"data:     {card.data_source}",
            f"trained:  {card.n_train} molecules, tested on {card.n_test}",
            f"calibrated: {calibrator.n_fitted} of {calibrator.n_labels} labels",
            "",
            "                          model     floor",
        ]
        for key, label in [
            ("macro_average_precision", "macro AP, experts"),
            ("macro_roc_auc", "macro AUC, experts"),
            ("consensus_macro_average_precision", "macro AP, consensus"),
        ]:
            lines.append(f"{label:<24} {self.metrics[key]:>7.4f}   {self.floor[key]:>7.4f}")
        return "\n".join(lines)


def train_odor_model_v2(
    *,
    name: str = DEFAULT_NAME,
    feature_set: str = DEFAULT_FEATURE_SET,
    seed: int = DEFAULT_SEED,
    c: float = DEFAULT_C,
    data: DatasetV2 | None = None,
) -> TrainingOutcomeV2:
    data = data if data is not None else build()
    split = data.split(seed=seed)
    features = feature_sets.build(feature_set, data.molecules)
    targets = data.targets(soft=True)

    # Train on everything but the test part; calibrate on out-of-fold predictions for
    # those same molecules. Calibrating on the validation part alone saw ~570 molecules
    # and left 48 descriptors with too few examples to calibrate (journal E-019).
    learn = np.concatenate([split.train, split.val])
    estimator = train_v2.fit(features.values, targets, learn, features.names, seed=seed, c=c)
    out_of_fold = _out_of_fold(features, targets, learn, data.groups, seed=seed, c=c)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        base_rates = np.nan_to_num(np.nanmean(targets[learn], axis=0), nan=0.0)
    calibrator = PerLabelCalibrator.fit(out_of_fold, targets[learn], base_rates=base_rates)

    test_scores = estimator.predict_proba(features.values[split.test])
    floor_scores = np.tile(base_rates, (split.test.size, 1))
    experts = data.targets(soft=False)[split.test]  # any yes is yes; not asked is unknown
    consensus = data.consensus()[split.test]
    metrics = _metrics(test_scores, experts, consensus)
    floor = _metrics(floor_scores, experts, consensus)
    metrics |= _calibration_check(calibrator, test_scores, targets[split.test])

    card = build_card(
        name=name,
        feature_set=feature_set,
        estimator="logistic-regression-ovr-blockwise-scaled, soft masked labels",
        seed=seed,
        label_names=data.label_names,
        data_source="pyrfume goodscents + leffingwell (dataset v2)",
        data_sha256=_combined_checksum(),
        split_strategy="stratified-by-skeleton",
        n_train=int(learn.size),
        n_test=int(split.test.size),
        metrics=metrics,
        baseline_metrics=floor,
        scope=data.scope,
    )
    model = OdorModel(
        card=card,
        estimator=estimator,
        calibrator=calibrator,
        scope=data.scope,
        reference=_reference_library(data, split),
    )
    return TrainingOutcomeV2(model=model, metrics=metrics, floor=floor)


FOLDS = 5


def _out_of_fold(features, targets, rows, groups, *, seed: int, c: float) -> np.ndarray:
    """Each row's prediction from a model trained without its fold.

    Folds are drawn by skeleton, so no isomer of a molecule is in the model that
    predicts it: the same leak the main split guards against.
    """
    rng = np.random.default_rng(seed)
    skeletons = sorted({groups[r] for r in rows})
    fold_of = dict(zip(skeletons, rng.integers(0, FOLDS, len(skeletons)), strict=True))
    fold = np.array([fold_of[groups[r]] for r in rows])
    predicted = np.zeros((rows.size, targets.shape[1]))
    for k in range(FOLDS):
        inside = rows[fold != k]
        model = train_v2.fit(features.values, targets, inside, features.names, seed=seed, c=c)
        predicted[fold == k] = model.predict_proba(features.values[rows[fold == k]])
    return predicted


def _calibration_check(calibrator, raw: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """The product's promise, measured: when it says "likely", how often is it right?

    On the test part, calibrated descriptors only, cells where experts were asked.
    Truth is the share of experts who said yes, the quantity the model is trained on.
    """
    probability = calibrator.transform(raw)
    usable = ~np.isnan(truth) & calibrator.fitted[None, :] & (probability >= 0.85)
    return {
        "calibration_likely_promised": float(probability[usable].mean()),
        "calibration_likely_observed": float(truth[usable].mean()),
        "calibration_likely_cells": float(usable.sum()),
    }


def _metrics(scores: np.ndarray, experts: np.ndarray, consensus: np.ndarray) -> dict[str, float]:
    on_experts = train_v2.score(scores, experts)
    on_consensus = train_v2.score(scores, consensus)
    known = (experts == 0) | (experts == 1)
    micro = train_v2.score(scores[known][:, None], experts[known][:, None])
    return {
        "macro_average_precision": on_experts["macro_ap"],
        "macro_roc_auc": on_experts["macro_auc"],
        "micro_average_precision": micro["macro_ap"],
        "micro_roc_auc": micro["macro_auc"],
        "labels_scored": float(on_experts["descriptors"]),
        "labels_total": float(experts.shape[1]),
        "consensus_macro_average_precision": on_consensus["macro_ap"],
        "consensus_macro_roc_auc": on_consensus["macro_auc"],
        "consensus_labels_scored": float(on_consensus["descriptors"]),
    }


def _reference_library(data: DatasetV2, split) -> tuple[ReferenceMolecule, ...]:
    """OpenPOM's records (MIT), each marked with the part of v2's split its skeleton is in.

    ``train`` means this model learnt from the molecule; ``unseen`` that v2 never had it.
    """
    part_of: dict[str, str] = {}
    for part, rows in split.parts().items():
        for row in rows:
            part_of[data.groups[row]] = part

    openpom = Dataset.load()
    openpom = openpom.restrict(openpom.in_scope(data.scope))
    names = np.asarray(openpom.label_names)
    library = []
    for index, molecule in enumerate(openpom.molecules):
        flat = Chem.MolToSmiles(molecule, isomericSmiles=False)
        library.append(
            ReferenceMolecule(
                smiles=flat,
                descriptors=tuple(names[openpom.labels[index] == 1]),
                split=part_of.get(flat, "unseen"),
            )
        )
    return tuple(library)


def _combined_checksum() -> str:
    """One fingerprint for the six pinned Pyrfume files the dataset is built from."""
    digest = hashlib.sha256()
    for source in sorted(PYRFUME_SOURCES, key=lambda s: s.name):
        digest.update(f"{source.name}:{source.sha256}\n".encode())
    return digest.hexdigest()
