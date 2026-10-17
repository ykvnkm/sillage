"""A molecule as the product sees it: an identity, not a graph.

The graph lives in RDKit, on the adapter side. The domain only needs what a user can
be shown and what two molecules can be compared by -- a canonical SMILES string.
"""

from __future__ import annotations

from dataclasses import dataclass

from sillage.domain.names import MoleculeNames


class InvalidMolecule(ValueError):
    """The input does not describe a molecule at all."""


@dataclass(frozen=True, slots=True)
class Molecule:
    """A parsed molecule: its identity, and the exact form a model reads."""

    smiles: str
    """Canonical SMILES **without stereochemistry** -- the molecule's identity for
    everything that looks it up: the reference library, the name directory, the map.
    Those sources record no stereo, so (R)- and (S)-carvone are one entry there."""

    input_smiles: str
    """Exactly what the caller sent, so the answer can be traced back to the question."""

    formula: str
    weight: float
    """Average molecular weight, Da."""

    isomeric_smiles: str = ""
    """Canonical SMILES **with** the stereochemistry the input gave; what a model is
    fed, since dataset v2 trains on isomeric structures (journal E-019, E-020). Equal
    to :attr:`smiles` when the input gave no stereo; empty in hand-built test objects,
    where it falls back to :attr:`smiles`."""

    @property
    def model_smiles(self) -> str:
        return self.isomeric_smiles or self.smiles

    @property
    def has_stereo(self) -> bool:
        """True when the input specified stereochemistry -- carvone is caraway on one
        side of the mirror and mint on the other, and the answer must say what the
        model did with that."""
        return self.model_smiles != self.smiles


@dataclass(frozen=True, slots=True)
class Neighbour:
    """A molecule from the dataset, with what experts said it smells of."""

    smiles: str
    similarity: float
    """Tanimoto on ECFP4, 0..1. Shown to orient the user, **not** as a confidence
    gate: measured on this dataset, nearest-neighbour similarity does not predict
    whether the model is right (journal E-008)."""

    descriptors: tuple[str, ...]
    """Expert labels from the dataset. Facts about this neighbour, not predictions."""

    names: MoleculeNames | None = None


@dataclass(frozen=True, slots=True)
class KnownMolecule:
    """What the dataset records about a molecule the user asked about."""

    descriptors: tuple[str, ...]
    seen_in_training: bool
    """True when the model was fitted on this very molecule. Then agreement between
    prediction and experts is recall, not evidence of skill -- and the product has
    to say so, or every famous molecule a user tries will look like proof."""
