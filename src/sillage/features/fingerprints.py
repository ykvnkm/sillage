"""Morgan (ECFP) fingerprints.

A Morgan fingerprint describes a molecule by the *substructures it contains*. For
every atom the algorithm looks at the neighbourhood within radius 0, 1, ... r,
hashes each of those little fragments into a number, and sets the corresponding
bit. Two molecules that share a fragment set the same bit, whatever the rest of
them looks like.

The price of a fixed length is collisions: different fragments can land on the
same bit, and the bit no longer identifies its fragment uniquely. Widening the
vector reduces collisions and costs memory; 2048 bits at radius 2 (the encoding
usually called ECFP4, because the *diameter* is 4) is the long-standing default in
cheminformatics and the one OpenPOM compares against.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from rdkit.Chem import Mol, rdFingerprintGenerator

from sillage.features.matrix import FeatureMatrix

DEFAULT_RADIUS = 2
DEFAULT_N_BITS = 2048


def morgan_fingerprints(
    molecules: Sequence[Mol],
    *,
    radius: int = DEFAULT_RADIUS,
    n_bits: int = DEFAULT_N_BITS,
    counts: bool = False,
) -> FeatureMatrix:
    """Encode molecules as Morgan fingerprints.

    Args:
        molecules: parsed RDKit molecules.
        radius: how far from each atom the fragments extend. 2 is ECFP4.
        n_bits: width of the vector. Larger means fewer hash collisions.
        counts: store how many times a fragment occurs instead of merely whether
            it occurs. Occasionally helps, and costs nothing to try.

    Returns:
        A matrix of shape ``(len(molecules), n_bits)``. Columns are named by bit
        index: a Morgan bit has no human-readable meaning, only an identity.
    """
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    encode = generator.GetCountFingerprintAsNumPy if counts else generator.GetFingerprintAsNumPy

    if not molecules:
        values = np.zeros((0, n_bits), dtype=np.uint8)
    else:
        values = np.vstack([encode(molecule) for molecule in molecules])

    prefix = "morgan_count" if counts else "morgan_bit"
    return FeatureMatrix(
        values=values.astype(np.float32),
        names=tuple(f"{prefix}_{index}" for index in range(n_bits)),
    )
