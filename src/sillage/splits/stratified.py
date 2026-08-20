"""Iterative stratification for multilabel data.

Ordinary stratification keeps the proportion of *one* class constant across the
parts. With 138 labels per row that is not possible: balancing one label unbalances
another, and a molecule carries four labels at once on average.

The algorithm of Sechidis, Tsoumakas and Vlahavas (2011) resolves the conflict by
being greedy in a specific order -- **the rarest label first**. At each step it
takes the label with the fewest examples still unassigned and places those examples
into the parts that are furthest from their quota for that label. Rare labels get
their choice while there is still freedom left; frequent labels are numerous enough
to even out afterwards.

That ordering is the whole idea, and it is exactly right for this dataset, where
``chamomile`` has 31 examples and ``fruity`` has 1902.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from sillage.splits.types import PARTS, Split

DEFAULT_PROPORTIONS = (0.8, 0.1, 0.1)


def iterative_stratification(
    labels: np.ndarray,
    proportions: Sequence[float] = DEFAULT_PROPORTIONS,
    *,
    seed: int = 0,
) -> Split:
    """Split rows so that every label keeps its proportions in every part.

    Args:
        labels: binary matrix of shape ``(n_rows, n_labels)``.
        proportions: fractions for train, val and test. Must sum to 1.
        seed: ties are broken at random; the seed makes that reproducible.

    Returns:
        A :class:`Split` over ``range(n_rows)``.
    """
    _validate(labels, proportions)
    rng = np.random.default_rng(seed)
    binary = labels.astype(bool)
    n_rows = binary.shape[0]

    # Quotas: how many rows each part still wants overall, and per label.
    wanted = np.asarray(proportions, dtype=float) * n_rows
    positives = binary.sum(axis=0).astype(float)
    wanted_per_label = np.outer(positives, proportions)

    unassigned = np.ones(n_rows, dtype=bool)
    left_per_label = positives.copy()
    assignment = np.full(n_rows, -1, dtype=np.int64)

    while unassigned.any():
        available = np.flatnonzero(left_per_label > 0)
        if available.size == 0:
            # Rows whose labels are all placed: nothing left to balance, so they
            # simply go wherever the overall quota is least satisfied.
            for row in np.flatnonzero(unassigned):
                part = _argmax_with_random_tiebreak(wanted, rng)
                assignment[row] = part
                wanted[part] -= 1
                unassigned[row] = False
            break

        rarest = available[np.argmin(left_per_label[available])]
        members = np.flatnonzero(unassigned & binary[:, rarest])
        rng.shuffle(members)

        for row in members:
            part = _pick_part(wanted_per_label[rarest], wanted, rng)
            assignment[row] = part
            unassigned[row] = False

            row_labels = np.flatnonzero(binary[row])
            wanted_per_label[row_labels, part] -= 1
            left_per_label[row_labels] -= 1
            wanted[part] -= 1

    return Split(
        **{name: np.flatnonzero(assignment == index) for index, name in enumerate(PARTS)},
        strategy="iterative-stratification",
        seed=seed,
    )


def _pick_part(label_quota: np.ndarray, overall_quota: np.ndarray, rng: np.random.Generator) -> int:
    """Choose the part that needs this label most; break ties on overall need.

    Both tiers matter. Without the first, rare labels pile up in one part. Without
    the second, the parts themselves drift away from the requested sizes.
    """
    candidates = np.flatnonzero(label_quota == label_quota.max())
    if candidates.size == 1:
        return int(candidates[0])

    restricted = overall_quota[candidates]
    best = candidates[np.flatnonzero(restricted == restricted.max())]
    return int(best[0] if best.size == 1 else rng.choice(best))


def _argmax_with_random_tiebreak(values: np.ndarray, rng: np.random.Generator) -> int:
    candidates = np.flatnonzero(values == values.max())
    return int(candidates[0] if candidates.size == 1 else rng.choice(candidates))


def _validate(labels: np.ndarray, proportions: Sequence[float]) -> None:
    if labels.ndim != 2:
        raise ValueError(f"Expected a 2-D label matrix, got shape {labels.shape}")
    if len(proportions) != len(PARTS):
        raise ValueError(f"Expected {len(PARTS)} proportions for {PARTS}, got {len(proportions)}")
    if not np.isclose(sum(proportions), 1.0):
        raise ValueError(f"Proportions must sum to 1, got {sum(proportions)}")
    if any(fraction <= 0 for fraction in proportions):
        raise ValueError(f"Every proportion must be positive, got {tuple(proportions)}")
