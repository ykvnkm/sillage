"""Named feature sets.

Naming them, rather than passing parameters around, means a results table can be
read a year later without reconstructing what "run 7" used. The name is the
experiment's identity and is what ends up in ``results/``.

The name is also what a saved model records in its card, which is why this module
lives under ``features`` and not under ``research``: the service has to rebuild
exactly the features the model was trained on, and it must never import research
code to do it (ADR-0001).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from rdkit.Chem import Mol

from sillage.features import morgan_fingerprints, rdkit_descriptors
from sillage.features.matrix import FeatureMatrix

FeatureSetBuilder = Callable[[Sequence[Mol]], FeatureMatrix]


def _fingerprints(radius: int, counts: bool) -> FeatureSetBuilder:
    def build(molecules: Sequence[Mol]) -> FeatureMatrix:
        return morgan_fingerprints(molecules, radius=radius, counts=counts)

    return build


def _fingerprints_with_descriptors(
    radius: int, counts: bool, chirality: bool = False
) -> FeatureSetBuilder:
    def build(molecules: Sequence[Mol]) -> FeatureMatrix:
        fingerprints = morgan_fingerprints(
            molecules, radius=radius, counts=counts, chirality=chirality
        )
        return fingerprints.hstack(rdkit_descriptors(molecules))

    return build


FEATURE_SETS: dict[str, FeatureSetBuilder] = {
    # Fingerprints alone: does the encoding of chain length matter (E-001)?
    "morgan-bin-r2": _fingerprints(2, counts=False),
    "morgan-cnt-r2": _fingerprints(2, counts=True),
    "morgan-bin-r3": _fingerprints(3, counts=False),
    "morgan-cnt-r3": _fingerprints(3, counts=True),
    # Physicochemistry alone: how far do 216 interpretable numbers get us?
    "descriptors": rdkit_descriptors,
    # Both, for every fingerprint variant.
    "morgan-bin-r2+desc": _fingerprints_with_descriptors(2, counts=False),
    "morgan-cnt-r2+desc": _fingerprints_with_descriptors(2, counts=True),
    "morgan-bin-r3+desc": _fingerprints_with_descriptors(3, counts=False),
    "morgan-cnt-r3+desc": _fingerprints_with_descriptors(3, counts=True),
    # Stereo-aware: needs isomeric input, which only dataset v2 has (E-019).
    "morgan-bin-r3-chiral+desc": _fingerprints_with_descriptors(3, counts=False, chirality=True),
}


def build(name: str, molecules: Sequence[Mol]) -> FeatureMatrix:
    """Build a named feature set, failing with the list of valid names."""
    try:
        builder = FEATURE_SETS[name]
    except KeyError:
        known = ", ".join(sorted(FEATURE_SETS))
        raise KeyError(f"Unknown feature set {name!r}. Known: {known}") from None
    return builder(molecules)
