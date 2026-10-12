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
    """A parsed molecule, reduced to what the model can actually distinguish."""

    smiles: str
    """Canonical SMILES **without stereochemistry**. Two inputs that differ only in
    stereo -- (R)- and (S)-carvone -- map to the same string, because the training
    data has no stereo either (the source column is ``nonStereoSMILES``) and the
    model cannot tell them apart. Pretending otherwise would be the worse option."""

    input_smiles: str
    """Exactly what the caller sent, so the answer can be traced back to the question."""

    formula: str
    weight: float
    """Average molecular weight, Da."""

    stereo_discarded: bool
    """True when the input carried stereo information that was dropped. The product
    must say so: carvone is caraway on one side of the mirror and mint on the other."""


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
