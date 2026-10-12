"""Nearest known molecules by fingerprint similarity."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from sillage.adapters.chem.reader import to_rdkit
from sillage.domain import KnownMolecule, Molecule, Neighbour
from sillage.features import morgan_fingerprints, parse_smiles
from sillage.models.applicability import tanimoto_similarity
from sillage.models.artifact import ReferenceMolecule

SIMILARITY_RADIUS = 2
"""ECFP4, the cheminformatics default for "how alike are these two structures".
Deliberately not the model's own radius-3 features: this answers a question for a
person, and should mean what the field means by similarity."""


class FingerprintReferenceLibrary:
    """Exact lookup by canonical SMILES, nearest neighbours by Tanimoto on ECFP4."""

    def __init__(self, entries: Sequence[ReferenceMolecule]) -> None:
        self._entries = tuple(entries)
        self._index_of = {entry.smiles: index for index, entry in enumerate(self._entries)}
        molecules = parse_smiles([entry.smiles for entry in self._entries])
        self._fingerprints = morgan_fingerprints(molecules, radius=SIMILARITY_RADIUS).values

    def __len__(self) -> int:
        return len(self._entries)

    def lookup(self, molecule: Molecule) -> KnownMolecule | None:
        index = self._index_of.get(molecule.smiles)
        if index is None:
            return None
        entry = self._entries[index]
        return KnownMolecule(descriptors=entry.descriptors, seen_in_training=entry.split == "train")

    def similarity_baseline(
        self, quantiles: tuple[float, ...] = (0.1, 0.25, 0.5), *, sample: int = 500
    ) -> dict[float, float]:
        """How similar an ordinary input is to its nearest known molecule.

        Drift monitoring watches the live distribution of exactly this number, and a
        live median of 0.45 means nothing without knowing that a dataset molecule's
        own nearest neighbour sits at, say, 0.55. Computed on a fixed sample of the
        library, each molecule excluded from its own neighbours as live lookups do.
        """
        n = len(self._entries)
        if n < 2:
            return {}
        rng = np.random.default_rng(0)
        rows = rng.choice(n, size=min(sample, n), replace=False)
        similarity = tanimoto_similarity(self._fingerprints[rows], self._fingerprints)
        similarity[np.arange(rows.size), rows] = -1.0
        nearest = similarity.max(axis=1)
        return {q: float(np.quantile(nearest, q)) for q in quantiles}

    def nearest(self, molecule: Molecule, k: int) -> tuple[Neighbour, ...]:
        if k <= 0 or not self._entries:
            return ()
        query = morgan_fingerprints([to_rdkit(molecule)], radius=SIMILARITY_RADIUS).values
        similarity = tanimoto_similarity(query, self._fingerprints)[0]

        # The molecule itself is reported by `lookup`, not as its own neighbour.
        itself = self._index_of.get(molecule.smiles)
        if itself is not None:
            similarity[itself] = -1.0

        # Ties broken by library order, so the same input always gives the same answer.
        order = np.argsort(-similarity, kind="stable")[:k]
        return tuple(
            Neighbour(
                smiles=self._entries[index].smiles,
                similarity=round(float(similarity[index]), 3),
                descriptors=self._entries[index].descriptors,
            )
            for index in order
            if similarity[index] >= 0.0
        )
