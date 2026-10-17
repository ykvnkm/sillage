"""Molecular graphs (journal E-020): stereo survives, spelling does not matter."""

from __future__ import annotations

import numpy as np
from rdkit import Chem

from sillage.features.graphs import (
    ATOM_FEATURES,
    ATOM_STEREO_COLUMNS,
    BOND_FEATURES,
    BOND_STEREO_COLUMNS,
    molecular_graph,
)

R_CARVONE = "CC1=CC[C@@H](CC1=O)C(C)=C"
S_CARVONE = "CC1=CC[C@H](CC1=O)C(C)=C"
FLAT_CARVONE = "CC1=CCC(CC1=O)C(C)=C"


def graph(smiles: str):
    return molecular_graph(Chem.MolFromSmiles(smiles))


def atom_multiset(smiles: str) -> list[tuple[float, ...]]:
    return sorted(map(tuple, graph(smiles).atoms.tolist()))


def test_shapes_follow_the_molecule() -> None:
    g = graph(R_CARVONE)

    assert g.atoms.shape == (11, ATOM_FEATURES)
    assert g.bond_features.shape == (11, BOND_FEATURES)
    assert g.bonds.shape == (11, 2)
    assert g.atoms.dtype == np.float32


def test_each_atom_has_exactly_one_value_per_one_hot_group() -> None:
    g = graph(R_CARVONE)

    assert np.all(g.atoms[:, :5].sum(axis=1) == 1)  # element
    assert np.all(g.bond_features[:, :5].sum(axis=1) == 1)  # bond type


def test_enantiomers_get_different_graphs() -> None:
    assert atom_multiset(R_CARVONE) != atom_multiset(S_CARVONE)


def test_unspecified_stereocentre_is_neither_enantiomer() -> None:
    flat = atom_multiset(FLAT_CARVONE)

    assert flat != atom_multiset(R_CARVONE)
    assert flat != atom_multiset(S_CARVONE)


def test_the_same_molecule_written_another_way_gives_the_same_graph() -> None:
    reordered = Chem.MolToSmiles(Chem.MolFromSmiles(R_CARVONE), doRandom=True, canonical=False)

    assert reordered != R_CARVONE
    assert atom_multiset(reordered) == atom_multiset(R_CARVONE)


def test_double_bond_geometry_is_kept() -> None:
    e = graph("CC(C)=CCC/C(C)=C/C=O").bond_features[:, -3:].sum(axis=0)
    z = graph("CC(C)=CCC/C(C)=C\\C=O").bond_features[:, -3:].sum(axis=0)

    assert e.tolist()[:2] == [1.0, 0.0]
    assert z.tolist()[:2] == [0.0, 1.0]


def test_the_input_molecule_is_left_untouched() -> None:
    molecule = Chem.MolFromSmiles(R_CARVONE)

    def props() -> list[dict]:
        return [
            {k: v for k, v in a.GetPropsAsDict(includePrivate=True).items() if k[:2] != "__"}
            for a in molecule.GetAtoms()
        ]

    before = props()
    molecular_graph(molecule)

    assert props() == before


def test_stereo_columns_hold_exactly_the_stereo_labels() -> None:
    """Blanking them must make the enantiomers, and the flat molecule, identical."""

    def blanked(smiles: str) -> list[tuple[float, ...]]:
        atoms = graph(smiles).atoms.copy()
        atoms[:, ATOM_STEREO_COLUMNS] = 0.0
        return sorted(map(tuple, atoms.tolist()))

    assert blanked(R_CARVONE) == blanked(S_CARVONE) == blanked(FLAT_CARVONE)
    bonds = graph("CC(C)=CCC/C(C)=C/C=O").bond_features
    assert bonds[:, BOND_STEREO_COLUMNS].tolist().count([1.0, 0.0, 0.0]) == 1
