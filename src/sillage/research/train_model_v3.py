"""Training the shipping ensemble: graph networks, fingerprints, descriptors (journal E-020).

Three components, each measured on its own and together on five paired splits:

  linear     logistic regression on Morgan bits + descriptors -- odor-baseline-v2
  boosting   histogram boosting on the 216 descriptors
  graph      five D-MPNNs from different seeds, stereo-aware

Their plain mean beat odor-baseline-v2 on all five splits against experts' consensus
(+0.029 macro AP) and against each collection (+0.034 Leffingwell, +0.030 Goodscents).

The linear and boosting parts learn from the training and validation parts together,
as v2 did. The graph networks learn from the training part and stop on the
validation part -- the recipe that was measured; a network trained without a held-out
part has nothing to tell it when to stop.

Calibration is fitted on out-of-fold predictions of the whole ensemble over those
same molecules, five folds by skeleton, each fold with its own three components; the
networks in a fold stop on a tenth of that fold's training skeletons. Six independent
jobs -- five folds and the final model -- run in parallel processes.
"""

from __future__ import annotations

import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace

import joblib
import numpy as np

from sillage.features import sets as feature_sets
from sillage.features.graphs import molecular_graph
from sillage.models import PerLabelCalibrator
from sillage.models.artifact import OdorModel, build_card
from sillage.models.graph_network import GraphEnsemble
from sillage.paths import project_root
from sillage.research import train_v2
from sillage.research.dataset_v2 import build
from sillage.research.train_model_v2 import (
    FOLDS,
    TrainingOutcomeV2,
    _calibration_check,
    _combined_checksum,
    _metrics,
    _reference_library,
)

DEFAULT_NAME = "odor-ensemble-v3"
LINEAR_FEATURES = "morgan-bin-r3+desc"
BOOSTING_FEATURES = "descriptors"
MEMBERS = 5
WORKERS = 3
STOP_SHARE = 0.1

# Chosen on validation only (scripts/run_gnn_tuning.py, results/gnn_tuning.csv).
GNN_CONFIG = dict(hidden=600, learning_rate=2e-3, epochs=100, patience=25)


@dataclass(frozen=True, slots=True)
class _Components:
    linear: object
    boosting: object
    graph_states: list[dict]
    depth: int


def train_odor_ensemble(*, name: str = DEFAULT_NAME, seed: int = 0) -> TrainingOutcomeV2:
    data = build()
    split = data.split(seed=seed)
    targets = data.targets(soft=True)
    learn = np.concatenate([split.train, split.val])
    folds = _folds(learn, data.groups, seed)

    jobs = [("final", split.train, split.val, None, seed)]
    for k in range(FOLDS):
        inside = learn[folds != k]
        fit_rows, stop_rows = _hold_out_skeletons(inside, data.groups, seed + k)
        jobs.append((k, fit_rows, stop_rows, learn[folds == k], seed))

    threads = max(1, (os.cpu_count() or 4) // WORKERS)
    with ProcessPoolExecutor(WORKERS, initializer=_limit_threads, initargs=(threads,)) as pool:
        results = dict(pool.map(_job, jobs))

    out_of_fold = np.zeros((learn.size, targets.shape[1]))
    for k in range(FOLDS):
        out_of_fold[folds == k] = results[k]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        base_rates = np.nan_to_num(np.nanmean(targets[learn], axis=0), nan=0.0)
    calibrator = PerLabelCalibrator.fit(out_of_fold, targets[learn], base_rates=base_rates)

    final: _Components = results["final"]
    graph = GraphEnsemble.from_state_dicts(final.graph_states, [final.depth] * MEMBERS)
    card = build_card(
        name=name,
        feature_set=LINEAR_FEATURES,
        estimator=(
            "mean of: logistic regression (Morgan r3 + descriptors), histogram boosting "
            f"(descriptors), {MEMBERS} D-MPNNs (stereo-aware graphs); soft masked labels"
        ),
        seed=seed,
        label_names=data.label_names,
        data_source="pyrfume goodscents + leffingwell (dataset v2)",
        data_sha256=_combined_checksum(),
        split_strategy="stratified-by-skeleton",
        n_train=int(learn.size),
        n_test=int(split.test.size),
        metrics={},
        baseline_metrics={},
        scope=data.scope,
        components=("linear", "boosting", "graph"),
        boosting_feature_set=BOOSTING_FEATURES,
    )
    model = OdorModel(
        card=card,
        estimator=final.linear,
        boosting=final.boosting,
        graph=graph,
        calibrator=calibrator,
        scope=data.scope,
        reference=_reference_library(data, split),
    )

    # Measured through the very scorer the service runs: numpy graph networks included.
    test_scores = model.scorer().raw([data.molecules[i] for i in split.test])
    floor_scores = np.tile(base_rates, (split.test.size, 1))
    experts = data.targets(soft=False)[split.test]
    consensus = data.consensus()[split.test]
    metrics = _metrics(test_scores, experts, consensus)
    floor = _metrics(floor_scores, experts, consensus)
    metrics |= _calibration_check(calibrator, test_scores, targets[split.test])
    model = replace(model, card=replace(card, metrics=metrics, baseline_metrics=floor))
    return TrainingOutcomeV2(model=model, metrics=metrics, floor=floor)


def _job(job):
    """One fold (or the final model), cached on disk: a failure after training --
    in calibration, scoring or saving -- must not cost another forty minutes."""
    key = job[0]
    cache = project_root() / "data" / "interim" / f"ensemble_job_{key}_seed{job[4]}.joblib"
    if cache.exists():
        return joblib.load(cache)
    result = _fit_job(job)
    cache.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result, cache)
    return result


def _fit_job(job):
    """Fit all three components; predict the held-out rows if there are any."""
    key, fit_rows, stop_rows, predict_rows, seed = job
    warnings.filterwarnings("ignore")
    from sillage.research import gnn

    data = build()
    targets = data.targets(soft=True)
    experts = data.targets(soft=False)
    linear_x = feature_sets.build(LINEAR_FEATURES, data.molecules)
    boosting_x = feature_sets.build(BOOSTING_FEATURES, data.molecules)
    graphs = [molecular_graph(m) for m in data.molecules]

    # Linear and boosting learn from everything but the predicted rows, stop set included.
    rows = np.concatenate([fit_rows, stop_rows])
    linear = train_v2.fit(linear_x.values, targets, rows, linear_x.names, seed=seed)
    boosting = train_v2.fit_boosting(boosting_x, targets, rows, seed=seed)
    config = gnn.TrainingConfig(**GNN_CONFIG)
    states = []
    for member in range(MEMBERS):
        trained = gnn.train(
            graphs, targets, fit_rows, stop_rows, experts[stop_rows],
            config=config, seed=seed * 100 + member,
        )  # fmt: skip
        states.append({k: v.detach().cpu().numpy() for k, v in trained.model.state_dict().items()})

    if predict_rows is None:
        return key, _Components(linear, boosting, states, config.depth)
    graph = GraphEnsemble.from_state_dicts(states, [config.depth] * MEMBERS)
    predicted = np.mean(
        [
            linear.predict_proba(linear_x.values[predict_rows]),
            boosting.predict_proba(boosting_x.values[predict_rows]),
            np.stack([graph.predict(graphs[i]) for i in predict_rows]),
        ],
        axis=0,
    )
    return key, predicted


def _folds(rows: np.ndarray, groups, seed: int) -> np.ndarray:
    """Fold of each row, drawn by skeleton -- the same draw as odor-baseline-v2's."""
    rng = np.random.default_rng(seed)
    skeletons = sorted({groups[r] for r in rows})
    fold_of = dict(zip(skeletons, rng.integers(0, FOLDS, len(skeletons)), strict=True))
    return np.array([fold_of[groups[r]] for r in rows])


def _hold_out_skeletons(rows: np.ndarray, groups, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split *rows* into fit and stop parts, a tenth of the skeletons in the latter."""
    rng = np.random.default_rng(seed)
    skeletons = sorted({groups[r] for r in rows})
    stop = set(rng.choice(skeletons, int(len(skeletons) * STOP_SHARE), replace=False))
    held = np.array([groups[r] in stop for r in rows])
    return rows[~held], rows[held]


_LIMITS = []


def _limit_threads(threads: int) -> None:
    """Share the cores between the parallel jobs: torch and boosting's OpenMP alike.
    Without it each of the three jobs' boosting takes every core, and they thrash."""
    import torch
    from threadpoolctl import threadpool_limits

    torch.set_num_threads(threads)
    _LIMITS.append(threadpool_limits(limits=threads))  # held for the process's life
