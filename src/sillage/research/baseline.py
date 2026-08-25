"""Running the phase 1 baseline grid.

Deliberately procedural. This is research code by the boundary drawn in ADR-0001:
it exists to be rewritten, it has one consumer, and layering it would slow down the
only thing that matters here -- changing a question and re-running.

The one discipline it does keep is that a run is fully described by its
:class:`RunSpec`. Model, feature set, split strategy and seed all end up in the
results table, so a number can always be traced back to what produced it.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from rdkit.Chem import Mol

from sillage.data.loaders import load_curated_openpom
from sillage.data.schemas import OPENPOM_DESCRIPTORS, SMILES_COLUMN
from sillage.features import parse_smiles
from sillage.features.matrix import FeatureMatrix
from sillage.metrics import MultilabelReport, evaluate
from sillage.models import (
    MultilabelClassifier,
    PrevalenceBaseline,
    gradient_boosting,
    logistic_regression,
    logistic_regression_single_scaler,
)
from sillage.research import feature_sets
from sillage.splits import Split, iterative_stratification, scaffold_split

# A factory takes the seed and the feature-column names: the block-wise scaler in
# `logistic_regression` needs to know which columns are fingerprints (journal E-005).
MODELS: dict[str, Callable[[int, tuple[str, ...]], object]] = {
    "prevalence": lambda seed, names: PrevalenceBaseline(),
    "logreg": lambda seed, names: MultilabelClassifier(
        lambda: logistic_regression(names, seed=seed)
    ),
    "logreg-std": lambda seed, names: MultilabelClassifier(
        lambda: logistic_regression_single_scaler(names, scaler="standard", seed=seed)
    ),
    "logreg-maxabs": lambda seed, names: MultilabelClassifier(
        lambda: logistic_regression_single_scaler(names, scaler="maxabs", seed=seed)
    ),
    "boosting": lambda seed, names: MultilabelClassifier(lambda: gradient_boosting(seed=seed)),
}

SPLIT_STRATEGIES = ("stratified", "scaffold")


@dataclass(frozen=True, slots=True)
class RunSpec:
    """Everything needed to reproduce one number in the results table."""

    model: str
    features: str
    split: str
    seed: int

    def __str__(self) -> str:
        return f"{self.model:<10} {self.features:<20} {self.split:<11} seed={self.seed}"


@dataclass(frozen=True, slots=True)
class Dataset:
    """The dataset, parsed once and reused by every run."""

    molecules: list[Mol]
    labels: np.ndarray
    label_names: tuple[str, ...]

    @classmethod
    def load(cls) -> Dataset:
        frame = load_curated_openpom().reset_index(drop=True)
        return cls(
            molecules=parse_smiles(frame[SMILES_COLUMN].tolist()),
            labels=frame[list(OPENPOM_DESCRIPTORS)].to_numpy(np.int8),
            label_names=OPENPOM_DESCRIPTORS,
        )

    def split(self, strategy: str, seed: int) -> Split:
        if strategy == "stratified":
            return iterative_stratification(self.labels, seed=seed)
        if strategy == "scaffold":
            # Grouping is deterministic; the seed only varies the model.
            return scaffold_split(self.molecules)
        raise ValueError(f"Unknown split strategy {strategy!r}; expected one of {SPLIT_STRATEGIES}")


def run_one(
    spec: RunSpec, data: Dataset, features: FeatureMatrix
) -> tuple[MultilabelReport, float]:
    """Train and evaluate a single configuration. Returns the report and elapsed seconds."""
    split = data.split(spec.split, spec.seed)
    model = MODELS[spec.model](spec.seed, features.names)
    values = features.values

    started = time.perf_counter()
    model.fit(values[split.train], data.labels[split.train])
    scores = model.predict_proba(values[split.test])
    elapsed = time.perf_counter() - started

    return evaluate(data.labels[split.test], scores, data.label_names), elapsed


def run_grid(
    specs: Iterable[RunSpec],
    *,
    data: Dataset | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[RunSpec, MultilabelReport]]:
    """Run every spec, caching feature matrices across runs that share one.

    Returns:
        A summary row per run, and the full per-label report for each run -- the
        latter is what step 1.6 needs to find where the model fails.
    """
    data = data if data is not None else Dataset.load()
    cache: dict[str, FeatureMatrix] = {}
    rows: list[dict[str, object]] = []
    reports: dict[RunSpec, MultilabelReport] = {}

    for spec in specs:
        if spec.features not in cache:
            cache[spec.features] = feature_sets.build(spec.features, data.molecules)

        report, elapsed = run_one(spec, data, cache[spec.features])
        reports[spec] = report
        rows.append(
            {
                "model": spec.model,
                "features": spec.features,
                "n_features": cache[spec.features].shape[1],
                "split": spec.split,
                "seed": spec.seed,
                "macro_ap": report.macro_average_precision,
                "macro_auc": report.macro_roc_auc,
                "micro_ap": report.micro_average_precision,
                "micro_auc": report.micro_roc_auc,
                "labels_scored": report.n_labels_scored,
                "seconds": elapsed,
            }
        )
        if verbose:
            print(f"{spec}  macro AP {report.macro_average_precision:.4f}  ({elapsed:5.1f}s)")

    return pd.DataFrame(rows), reports


def grid(
    models: Sequence[str],
    features: Sequence[str],
    splits: Sequence[str],
    seeds: Sequence[int],
) -> list[RunSpec]:
    """Cartesian product, ordered so that runs sharing a feature set stay adjacent."""
    return [
        RunSpec(model=model, features=feature, split=split, seed=seed)
        for feature in features
        for model in models
        for split in splits
        for seed in seeds
    ]


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Mean and spread across seeds, which is the only honest way to read these runs.

    With three positive examples of the rarest descriptor in the test part, a single
    seed produces a number that moves under its own noise. A difference smaller than
    the spread is not a difference.
    """
    grouped = results.groupby(["model", "features", "split"], sort=False)
    summary = grouped.agg(
        macro_ap_mean=("macro_ap", "mean"),
        macro_ap_std=("macro_ap", "std"),
        macro_auc_mean=("macro_auc", "mean"),
        micro_ap_mean=("micro_ap", "mean"),
        micro_auc_mean=("micro_auc", "mean"),
        labels_scored=("labels_scored", "min"),
        n_seeds=("seed", "nunique"),
        seconds=("seconds", "mean"),
    )
    return summary.fillna({"macro_ap_std": 0.0}).sort_values("macro_ap_mean", ascending=False)
