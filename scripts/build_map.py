"""Build the odour map: every reference molecule placed by how the model thinks it smells.

    uv run python scripts/build_map.py            # writes models/<name>/map.npz

Each molecule is embedded by the model -- for the baseline, its 138 predicted
descriptor probabilities; for the GNN of phase 2, its penultimate layer -- and UMAP
lays that space out in two dimensions. Positions are the model's view; colours (the
odour family) come from what experts recorded, so the map shows where the two agree.

UMAP runs here, once, and never in the service: it would pull numba and LLVM into the
image and take a minute to warm up on a small CPU. A new molecule is placed instead by
its nearest reference molecules in the same embedding space
(:class:`sillage.adapters.models.odor_map.EmbeddingMap`). This script measures how far
that shortcut lands from UMAP's own placement, and prints it.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import umap

from sillage.adapters.models.odor_map import MAP_FILE, EmbeddingMap
from sillage.domain.families import Family, dominant_family
from sillage.features import sets as feature_sets
from sillage.models.artifact import OdorModel
from sillage.models.compiled import LinearMultilabel
from sillage.paths import DEFAULT_MODEL, project_root

SEED = 0
UMAP_PARAMS = {"n_neighbors": 30, "min_dist": 0.3, "metric": "cosine", "random_state": SEED}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default=DEFAULT_MODEL)
    args = parser.parse_args()

    directory = project_root() / "models" / args.name
    model = OdorModel.load(directory)
    reference = model.reference

    print(f"embedding {len(reference)} reference molecules", file=sys.stderr)
    features = feature_sets.build(model.card.feature_set, model.reference_molecules())
    embeddings = LinearMultilabel.from_classifier(model.estimator).predict_proba(features.values)

    frequency = dict(zip(model.label_names, model.calibrator.base_rates, strict=True))
    families = np.array(
        [list(Family).index(dominant_family(r.descriptors, frequency)) for r in reference],
        dtype=np.int8,
    )

    print("fitting UMAP", file=sys.stderr)
    coords = normalise(umap.UMAP(**UMAP_PARAMS).fit_transform(embeddings))

    report_placement(embeddings)
    report_families(coords, families)

    np.savez_compressed(
        directory / MAP_FILE,
        smiles=np.array([r.smiles for r in reference]),
        coords=coords.astype(np.float32),
        embeddings=embeddings.astype(np.float16),
        families=families,
        family_order=np.array([f.value for f in Family]),
    )
    print(f"wrote {(directory / MAP_FILE).relative_to(project_root())}")
    return 0


def normalise(coords: np.ndarray) -> np.ndarray:
    """Scale into [0, 1] on both axes, keeping the aspect ratio."""
    shifted = coords - coords.min(axis=0)
    return shifted / shifted.max()


def report_placement(embeddings: np.ndarray) -> None:
    """How far nearest-neighbour placement lands from UMAP's own ``transform``.

    10% of the molecules are held out, UMAP is fitted on the rest, and each held-out
    molecule is placed both ways. The distance is in map units (the map spans 0..1).
    """
    rng = np.random.default_rng(SEED)
    held_out = rng.random(len(embeddings)) < 0.1
    fit, rest = embeddings[~held_out], embeddings[held_out]

    reducer = umap.UMAP(**UMAP_PARAMS).fit(fit)
    raw = reducer.embedding_
    low, span = raw.min(axis=0), (raw - raw.min(axis=0)).max()
    coords = (raw - low) / span
    umap_placed = (reducer.transform(rest) - low) / span

    shortcut = EmbeddingMap(
        smiles=np.array([str(i) for i in range(len(fit))]),
        coords=coords,
        embeddings=fit,
        families=np.zeros(len(fit), dtype=np.int8),
        family_order=np.array([f.value for f in Family]),
    )
    knn_placed = np.array([shortcut.locate_vector(e) for e in rest])
    error = np.linalg.norm(knn_placed - umap_placed, axis=1)
    print(
        f"placement of {len(rest)} held-out molecules, nearest neighbours vs UMAP transform: "
        f"median {np.median(error):.3f}, 90th percentile {np.quantile(error, 0.9):.3f} "
        "(map spans 0..1)"
    )


def report_families(coords: np.ndarray, families: np.ndarray, k: int = 15) -> None:
    """Does the model's odour space keep experts' families together?

    For every molecule with a named family: the share of its k nearest neighbours *on
    the map* that share that family, against the same share when positions are
    shuffled, which is just the family's frequency.
    """
    named = families != list(Family).index(Family.OTHER)
    distances = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    nearest = np.argsort(distances, axis=1)[:, :k]

    print(f"{'family':<12} {'n':>5} {'neighbours same family':>24} {'by chance':>10}")
    for index, family in enumerate(Family):
        if family is Family.OTHER:
            continue
        members = np.flatnonzero(families == index)
        if members.size == 0:
            continue
        same = (families[nearest[members]] == index).mean()
        chance = (families[named] == index).mean()
        print(f"{family.value:<12} {members.size:>5} {same:>24.2f} {chance:>10.2f}")


if __name__ == "__main__":
    raise SystemExit(main())
