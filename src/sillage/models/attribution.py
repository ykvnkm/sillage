"""Which atoms make the model hear a note: occlusion, one atom at a time (journal E-021).

An atom's contribution is how much the note's calibrated probability falls when that
atom is taken away. "Taken away" means the same thing to every component, each in
its own representation:

* **linear** (Morgan bits + descriptors): the bits whose every occurrence involves the
  atom are switched off -- the similarity maps of Riniker and Landrum (2013). A bit
  also set by an environment elsewhere in the molecule stays on, since removing this
  atom would not remove that fragment.
* **graph networks**: the atom and its bonds are deleted from the graph.
* **boosting**: works on whole-molecule descriptors -- weight, polarity, volatility --
  which no single atom owns. It is held fixed, so its share of the answer is never
  attributed to an atom, and the product says so instead of spreading it around.

Contributions are in probability points of the calibrated output: +0.20 means "with
this atom the model says 20 points more". Occlusion rather than gradients because it
is the same question for every component, needs no torch, and its answer is a
quantity a user can read.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rdkit import Chem
from rdkit.Chem import Mol, rdFingerprintGenerator

from sillage.features import sets as feature_sets
from sillage.features.graphs import MolecularGraph, molecular_graph
from sillage.models.calibration import PerLabelCalibrator

# The fingerprint half of "morgan-bin-r3+desc": the only linear feature set an
# attributable model uses. Anything else is refused rather than approximated.
FINGERPRINT_SETS = {"morgan-bin-r3+desc": (3, 2048)}


class NotAttributable(ValueError):
    """The model's linear features are not ones whose bits can be traced to atoms."""


@dataclass(frozen=True, slots=True)
class AtomContributions:
    """Per-atom change in one note's probability, in the input molecule's atom order."""

    label: int
    probability: float
    atoms: np.ndarray


def explain(
    scorer, calibrator: PerLabelCalibrator, molecule: Mol, labels: list[int]
) -> list[AtomContributions]:
    """Contributions of every atom of *molecule* to each of *labels*."""
    if scorer.feature_set not in FINGERPRINT_SETS:
        raise NotAttributable(f"cannot trace feature set {scorer.feature_set!r} to atoms")
    radius, n_bits = FINGERPRINT_SETS[scorer.feature_set]
    n_atoms = molecule.GetNumAtoms()

    linear_x = feature_sets.build(scorer.feature_set, [molecule]).values[0]
    occluded_x = np.repeat(linear_x[None, :], n_atoms, axis=0)
    for atom, bits in enumerate(_bits_owned_by_each_atom(molecule, radius, n_bits)):
        occluded_x[atom, bits] = 0.0
    parts_full = [scorer.linear.predict_proba(linear_x[None, :])[0]]
    parts_occluded = [scorer.linear.predict_proba(occluded_x)]

    if scorer.boosting is not None:
        boost = scorer.boosting.predict_proba(
            feature_sets.build(scorer.boosting_feature_set, [molecule]).values
        )[0]
        parts_full.append(boost)
        parts_occluded.append(np.repeat(boost[None, :], n_atoms, axis=0))

    if scorer.graph is not None:
        graph = molecular_graph(molecule)
        parts_full.append(scorer.graph.predict(graph))
        parts_occluded.append(
            np.stack([scorer.graph.predict(without_atom(graph, a)) for a in range(n_atoms)])
        )

    full = calibrator.transform(np.mean(parts_full, axis=0)[None, :])[0]
    occluded = calibrator.transform(np.mean(parts_occluded, axis=0))
    return [
        AtomContributions(label=j, probability=float(full[j]), atoms=full[j] - occluded[:, j])
        for j in labels
    ]


def without_atom(graph: MolecularGraph, atom: int) -> MolecularGraph:
    """The graph with one atom and every bond touching it removed."""
    keep_atoms = np.arange(graph.n_atoms) != atom
    keep_bonds = (graph.bonds != atom).all(axis=1)
    new_index = np.cumsum(keep_atoms) - 1
    return MolecularGraph(
        atoms=graph.atoms[keep_atoms],
        bond_features=graph.bond_features[keep_bonds],
        bonds=new_index[graph.bonds[keep_bonds]],
    )


def _bits_owned_by_each_atom(molecule: Mol, radius: int, n_bits: int) -> list[list[int]]:
    """For each atom, the set bits that would vanish without it.

    A bit vanishes only if every environment that sets it contains the atom.
    """
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    output = rdFingerprintGenerator.AdditionalOutput()
    output.AllocateBitInfoMap()
    generator.GetFingerprintAsNumPy(molecule, additionalOutput=output)

    owned: list[list[int]] = [[] for _ in range(molecule.GetNumAtoms())]
    for bit, environments in output.GetBitInfoMap().items():
        shared = None
        for centre, env_radius in environments:
            atoms = _environment_atoms(molecule, centre, env_radius)
            shared = atoms if shared is None else shared & atoms
        for atom in shared or ():
            owned[atom].append(bit)
    return owned


def _environment_atoms(molecule: Mol, centre: int, radius: int) -> set[int]:
    if radius == 0:
        return {centre}
    atoms = {centre}
    for bond in Chem.FindAtomEnvironmentOfRadiusN(molecule, radius, centre):
        b = molecule.GetBondWithIdx(bond)
        atoms.update((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
    return atoms
