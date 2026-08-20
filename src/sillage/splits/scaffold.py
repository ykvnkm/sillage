"""Scaffold split: structurally similar molecules never straddle the parts.

A stratified split answers "how well does the model do on molecules like the ones
it saw". A scaffold split answers the harder question: "how well does it do on
molecules unlike anything it saw". The second number is always worse and always
closer to what happens when someone types a new SMILES into the service.

The grouping is a hybrid, and the reason is measured rather than assumed.

The textbook key is the Bemis-Murcko scaffold: strip side chains, keep the ring
system and the linkers between rings. On this dataset it degenerates -- 45% of the
molecules have no rings at all, so they share the empty scaffold and land in a
single group covering nearly half the data.

Acyclic molecules therefore get a different key: the set of their Morgan fragments.
That collapses homologous series -- heptanal, octanal and decanal produce identical
binary fingerprints -- which is exactly the leakage worth preventing here, since
those molecules are indistinguishable to the model but carry different labels.

Measured on the 4983 molecules: 2482 groups, largest 762 (15.3%, the benzene ring).
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
from rdkit.Chem import Mol
from rdkit.Chem.Scaffolds import MurckoScaffold

from sillage.features.fingerprints import morgan_fingerprints
from sillage.splits.types import PARTS, Split

DEFAULT_PROPORTIONS = (0.8, 0.1, 0.1)


def scaffold_keys(molecules: Sequence[Mol]) -> list[str]:
    """Group key for every molecule: ring scaffold, or fragment set when acyclic."""
    if not molecules:
        return []

    scaffolds = [
        MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=False)
        for molecule in molecules
    ]
    fingerprints = morgan_fingerprints(molecules).values.astype(np.uint8)

    return [
        scaffold or f"acyclic:{hashlib.sha1(fingerprints[i].tobytes()).hexdigest()}"
        for i, scaffold in enumerate(scaffolds)
    ]


def scaffold_split(
    molecules: Sequence[Mol],
    proportions: Sequence[float] = DEFAULT_PROPORTIONS,
) -> Split:
    """Split molecules so that no structural group is shared between parts."""
    return group_split(scaffold_keys(molecules), proportions)


def group_split(keys: Sequence[str], proportions: Sequence[float] = DEFAULT_PROPORTIONS) -> Split:
    """Assign whole groups to parts, largest group first.

    Deterministic by construction, with no seed: groups are ordered by size and
    then by key, and each one goes to whichever part is furthest below its target.
    Largest-first matters -- placing a 762-molecule group after the small ones would
    overshoot whichever part received it.
    """
    if len(proportions) != len(PARTS):
        raise ValueError(f"Expected {len(PARTS)} proportions for {PARTS}, got {len(proportions)}")
    if not np.isclose(sum(proportions), 1.0):
        raise ValueError(f"Proportions must sum to 1, got {sum(proportions)}")

    members: dict[str, list[int]] = {}
    for index, key in enumerate(keys):
        members.setdefault(key, []).append(index)

    n_rows = len(keys)
    targets = np.asarray(proportions, dtype=float) * n_rows
    filled = np.zeros(len(PARTS), dtype=float)
    assignment = np.full(n_rows, -1, dtype=np.int64)

    ordered = sorted(members.items(), key=lambda item: (-len(item[1]), item[0]))
    for _, rows in ordered:
        part = int(np.argmax(targets - filled))
        assignment[rows] = part
        filled[part] += len(rows)

    return Split(
        **{name: np.flatnonzero(assignment == index) for index, name in enumerate(PARTS)},
        strategy="scaffold",
        seed=0,
    )
