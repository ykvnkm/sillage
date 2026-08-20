"""Properties both splitters must hold, plus the one that justifies stratification."""

from __future__ import annotations

import numpy as np
import pytest

from sillage.features import parse_smiles
from sillage.splits import (
    Split,
    group_split,
    iterative_stratification,
    label_coverage,
    scaffold_keys,
    scaffold_split,
)


def make_labels(n_rows: int = 300, n_labels: int = 12, seed: int = 0) -> np.ndarray:
    """Labels with a deliberately awful imbalance, like the real dataset."""
    rng = np.random.default_rng(seed)
    frequencies = np.geomspace(0.6, 0.02, n_labels)
    labels = (rng.random((n_rows, n_labels)) < frequencies).astype(np.int8)
    empty = labels.sum(axis=1) == 0
    labels[empty, rng.integers(0, n_labels, empty.sum())] = 1
    return labels


# --- the container -----------------------------------------------------------


def test_split_rejects_overlapping_parts() -> None:
    """A row in both train and test is leakage in its purest form."""
    with pytest.raises(ValueError, match="overlap"):
        Split(
            train=np.array([0, 1]),
            val=np.array([1]),
            test=np.array([2]),
            strategy="broken",
            seed=0,
        )


def test_split_rejects_missing_rows() -> None:
    """Rows silently dropped would make the metrics describe a smaller dataset."""
    with pytest.raises(ValueError, match="cover"):
        Split(
            train=np.array([0]),
            val=np.array([1]),
            test=np.array([5]),
            strategy="broken",
            seed=0,
        )


# --- iterative stratification ------------------------------------------------


def test_every_row_is_used_exactly_once() -> None:
    labels = make_labels()

    split = iterative_stratification(labels)

    assert split.n_rows == len(labels)


def test_part_sizes_are_close_to_the_requested_proportions() -> None:
    labels = make_labels(n_rows=1000)

    sizes = iterative_stratification(labels, (0.8, 0.1, 0.1)).sizes

    assert sizes["train"] == pytest.approx(800, abs=25)
    assert sizes["val"] == pytest.approx(100, abs=25)
    assert sizes["test"] == pytest.approx(100, abs=25)


def test_the_same_seed_gives_the_same_split() -> None:
    labels = make_labels()

    first = iterative_stratification(labels, seed=7)
    second = iterative_stratification(labels, seed=7)

    np.testing.assert_array_equal(first.test, second.test)


def test_a_different_seed_gives_a_different_split() -> None:
    """Needed for the multi-seed reporting that rare labels require."""
    labels = make_labels()

    assert not np.array_equal(
        iterative_stratification(labels, seed=1).test,
        iterative_stratification(labels, seed=2).test,
    )


def test_stratification_keeps_rare_labels_present_where_random_loses_them() -> None:
    """The reason the algorithm exists, stated as a comparison.

    A random split leaves some rare label with zero positives in a part, and its
    average precision there is undefined. Stratification is expected to do better
    across seeds -- that is the whole claim, so it is worth asserting.
    """
    labels = make_labels(n_rows=400, n_labels=20)
    rng = np.random.default_rng(0)

    stratified_gaps = 0
    random_gaps = 0
    for seed in range(10):
        stratified_gaps += int(
            (label_coverage(labels, iterative_stratification(labels, seed=seed)) == 0).sum()
        )

        shuffled = rng.permutation(len(labels))
        bounds = (int(0.8 * len(labels)), int(0.9 * len(labels)))
        random_split = Split(
            train=np.sort(shuffled[: bounds[0]]),
            val=np.sort(shuffled[bounds[0] : bounds[1]]),
            test=np.sort(shuffled[bounds[1] :]),
            strategy="random",
            seed=seed,
        )
        random_gaps += int((label_coverage(labels, random_split) == 0).sum())

    assert stratified_gaps < random_gaps


def test_proportions_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        iterative_stratification(make_labels(), (0.5, 0.2, 0.2))


def test_label_coverage_counts_positives_per_part() -> None:
    labels = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.int8)
    split = Split(
        train=np.array([0, 1]), val=np.array([2]), test=np.array([3]), strategy="manual", seed=0
    )

    np.testing.assert_array_equal(label_coverage(labels, split), [[2, 0, 0], [0, 1, 1]])


# --- scaffold split ----------------------------------------------------------

CYCLIC = ["c1ccccc1C", "c1ccccc1CC", "c1ccccc1CCC", "C1CCCCC1C", "C1CCCCC1CC"]

# Heptanal, octanal, decanal. Long enough that every radius-2 environment has
# already appeared, so another CH2 adds no new fragment and the vectors coincide.
LONG_HOMOLOGS = ["CCCCCCC=O", "CCCCCCCC=O", "CCCCCCCCCC=O"]

# Pentanal and hexanal. Short chains still differ: near the ends, the radius-2
# neighbourhoods are not yet saturated.
SHORT_HOMOLOGS = ["CCCCC=O", "CCCCCC=O"]


def test_molecules_sharing_a_ring_scaffold_share_a_key() -> None:
    keys = scaffold_keys(parse_smiles(["c1ccccc1C", "c1ccccc1CCC"]))

    assert keys[0] == keys[1] == "c1ccccc1"


def test_acyclic_molecules_do_not_all_collapse_into_one_group() -> None:
    """The failure mode of a textbook scaffold split: an empty scaffold for 45% of rows."""
    keys = scaffold_keys(parse_smiles(["CCO", "CCCO", "CC(=O)O", "CSCC"]))

    assert all(key.startswith("acyclic:") for key in keys)
    assert len(set(keys)) > 1


def test_long_homologous_series_lands_in_a_single_group() -> None:
    """Aldehydes indistinguishable to the model must not straddle the split.

    Heptanal, octanal and decanal share a binary Morgan fingerprint while carrying
    different odour labels. Training on one and testing on another would score a
    memorised answer as generalisation.
    """
    keys = scaffold_keys(parse_smiles(LONG_HOMOLOGS))

    assert len(set(keys)) == 1


def test_short_homologues_remain_distinguishable() -> None:
    """Documents where the collapse starts, so the limitation is not mistaken for a bug."""
    keys = scaffold_keys(parse_smiles(SHORT_HOMOLOGS))

    assert len(set(keys)) == len(SHORT_HOMOLOGS)


def test_no_group_is_shared_between_parts() -> None:
    """The defining property: seeing a scaffold in training must not help in test."""
    keys = ["a"] * 40 + ["b"] * 30 + ["c"] * 20 + [f"solo{i}" for i in range(10)]

    split = group_split(keys)

    part_of = {name: set(indices.tolist()) for name, indices in split.parts().items()}
    for key in set(keys):
        rows = {index for index, value in enumerate(keys) if value == key}
        assert sum(bool(rows & part) for part in part_of.values()) == 1


def test_group_split_is_deterministic_without_a_seed() -> None:
    keys = ["a"] * 40 + ["b"] * 30 + ["c"] * 20 + [f"solo{i}" for i in range(10)]

    np.testing.assert_array_equal(group_split(keys).test, group_split(keys).test)


def test_scaffold_split_covers_every_molecule() -> None:
    split = scaffold_split(parse_smiles(CYCLIC + LONG_HOMOLOGS))

    assert split.n_rows == len(CYCLIC + LONG_HOMOLOGS)
    assert split.strategy == "scaffold"


def test_empty_input_produces_no_keys() -> None:
    assert scaffold_keys([]) == []
