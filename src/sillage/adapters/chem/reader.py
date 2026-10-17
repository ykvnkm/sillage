"""Reading user text into a domain :class:`Molecule` with RDKit."""

from __future__ import annotations

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Mol, rdMolDescriptors

from sillage.domain import InvalidMolecule, Molecule

RDLogger.DisableLog("rdApp.*")


class RdkitMoleculeReader:
    """Parses SMILES and reduces it to the form the model was trained on."""

    def read(self, text: str) -> Molecule:
        parsed = Chem.MolFromSmiles(text)
        if parsed is None or parsed.GetNumAtoms() == 0:
            raise InvalidMolecule(f"not a valid SMILES string: {text!r}")

        flat = Chem.MolToSmiles(parsed, isomericSmiles=False)
        return Molecule(
            smiles=flat,
            input_smiles=text,
            formula=rdMolDescriptors.CalcMolFormula(parsed),
            weight=float(Descriptors.MolWt(parsed)),
            isomeric_smiles=Chem.MolToSmiles(parsed),
        )


def to_rdkit(molecule: Molecule) -> Mol:
    """The RDKit graph behind a domain molecule, stereo included where the input gave it.

    Re-parsed from the canonical string rather than carried alongside it: the domain
    object stays free of RDKit, and parsing a hundred-character SMILES costs tens of
    microseconds -- nothing next to the 100 ms the estimator takes.
    """
    parsed = Chem.MolFromSmiles(molecule.model_smiles)
    if parsed is None:  # canonical output of RDKit itself; failing here is a bug
        raise RuntimeError(
            f"RDKit cannot re-read its own canonical SMILES {molecule.model_smiles!r}"
        )
    return parsed
