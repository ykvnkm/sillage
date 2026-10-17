"""Molecules as graphs: atoms are nodes, bonds are edges, each with a feature vector.

A fingerprint hashes fragments into a fixed number of bits and forgets where they sit
relative to each other; a graph network reads the molecule itself. It is also the only
representation here that keeps stereochemistry without loss. The fingerprint's
chirality flag gained nothing (journal E-019), while Goodscents describes the two
isomers differently in 300 of 331 pairs (E-015).

Stereo is encoded with CIP labels -- R/S on atoms, E/Z on bonds -- not with RDKit's
chiral tags. A chiral tag (clockwise / counter-clockwise) is relative to the order the
atoms happen to be written in, so the same enantiomer typed two ways would get two
different graphs. A CIP label belongs to the molecule, not to its spelling.

NumPy and RDKit only: the service will build these graphs for every request, and it
must not import torch to do it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rdkit import Chem
from rdkit.Chem import Mol, rdCIPLabeler

ELEMENTS = ("C", "N", "O", "S")
DEGREES = (0, 1, 2, 3, 4)
HYDROGENS = (0, 1, 2, 3)
CHARGES = (-1, 0, 1)
HYBRIDISATIONS = (
    Chem.HybridizationType.SP,
    Chem.HybridizationType.SP2,
    Chem.HybridizationType.SP3,
)
RING_SIZES = (3, 4, 5, 6, 7, 8)
ATOM_STEREO = ("R", "S")
BOND_TYPES = (
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
)
BOND_STEREO = ("E", "Z")


def _one_hot(value, choices) -> list[float]:
    """One slot per choice plus a last one for "anything else"."""
    out = [0.0] * (len(choices) + 1)
    out[choices.index(value) if value in choices else -1] = 1.0
    return out


ATOM_FEATURES = (
    len(ELEMENTS) + 1
    + len(DEGREES) + 1
    + len(HYDROGENS) + 1
    + len(CHARGES) + 1
    + len(HYBRIDISATIONS) + 1
    + 2  # aromatic, in a ring
    + len(RING_SIZES)
    + len(ATOM_STEREO) + 1  # R, S, or a stereocentre the input left unspecified
    + 1  # mass / 100
)  # fmt: skip
BOND_FEATURES = len(BOND_TYPES) + 1 + 2 + len(BOND_STEREO) + 1

# Where the stereo labels sit, for experiments that blank them out (journal E-020).
ATOM_STEREO_COLUMNS = slice(ATOM_FEATURES - 1 - (len(ATOM_STEREO) + 1), ATOM_FEATURES - 1)
BOND_STEREO_COLUMNS = slice(BOND_FEATURES - (len(BOND_STEREO) + 1), BOND_FEATURES)


@dataclass(frozen=True, slots=True)
class MolecularGraph:
    """One molecule. ``bonds`` lists each bond once, as a pair of atom indices."""

    atoms: np.ndarray  # (n_atoms, ATOM_FEATURES) float32
    bond_features: np.ndarray  # (n_bonds, BOND_FEATURES) float32
    bonds: np.ndarray  # (n_bonds, 2) int64

    @property
    def n_atoms(self) -> int:
        return int(self.atoms.shape[0])


def molecular_graph(molecule: Mol) -> MolecularGraph:
    """Featurise heavy atoms and the bonds between them, stereo included."""
    molecule = Chem.Mol(molecule)  # the labeller writes properties; leave the input alone
    rdCIPLabeler.AssignCIPLabels(molecule)
    unspecified = {
        index
        for index, label in Chem.FindMolChiralCenters(
            molecule, includeUnassigned=True, useLegacyImplementation=False
        )
        if label == "?"
    }
    ring_info = molecule.GetRingInfo()

    atoms = []
    for atom in molecule.GetAtoms():
        index = atom.GetIdx()
        cip = atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else None
        stereo = [0.0, 0.0, 0.0]
        if cip in ATOM_STEREO:
            stereo[ATOM_STEREO.index(cip)] = 1.0
        elif index in unspecified:
            stereo[-1] = 1.0
        atoms.append(
            _one_hot(atom.GetSymbol(), ELEMENTS)
            + _one_hot(atom.GetDegree(), DEGREES)
            + _one_hot(atom.GetTotalNumHs(), HYDROGENS)
            + _one_hot(atom.GetFormalCharge(), CHARGES)
            + _one_hot(atom.GetHybridization(), HYBRIDISATIONS)
            + [float(atom.GetIsAromatic()), float(atom.IsInRing())]
            + [float(ring_info.IsAtomInRingOfSize(index, size)) for size in RING_SIZES]
            + stereo
            + [atom.GetMass() / 100.0]
        )

    bond_features, bonds = [], []
    for bond in molecule.GetBonds():
        cip = bond.GetProp("_CIPCode") if bond.HasProp("_CIPCode") else None
        bond_features.append(
            [
                *_one_hot(bond.GetBondType(), BOND_TYPES),
                float(bond.GetIsConjugated()),
                float(bond.IsInRing()),
                *_one_hot(cip, BOND_STEREO),
            ]
        )
        bonds.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))

    return MolecularGraph(
        atoms=np.asarray(atoms, dtype=np.float32).reshape(-1, ATOM_FEATURES),
        bond_features=np.asarray(bond_features, dtype=np.float32).reshape(-1, BOND_FEATURES),
        bonds=np.asarray(bonds, dtype=np.int64).reshape(-1, 2),
    )
