"""Parsing SMILES into RDKit molecules."""

from __future__ import annotations

from collections.abc import Sequence

from rdkit import Chem, RDLogger
from rdkit.Chem import Mol

# RDKit reports parse failures on stderr and keeps going. We detect failures from
# the return value instead and raise, so silencing the stream loses no information.
RDLogger.DisableLog("rdApp.*")


class InvalidSmilesError(ValueError):
    """Raised when a SMILES string cannot be parsed.

    Failing loudly is deliberate. RDKit returns ``None`` for unparseable input, and
    a silent ``None`` propagates as a dropped row -- so the model trains on fewer
    molecules than the metrics claim, and nothing anywhere says so.
    """

    def __init__(self, failures: Sequence[tuple[int, str]]) -> None:
        preview = ", ".join(f"row {index}: {smiles!r}" for index, smiles in failures[:5])
        suffix = f" (and {len(failures) - 5} more)" if len(failures) > 5 else ""
        super().__init__(f"{len(failures)} SMILES could not be parsed: {preview}{suffix}")
        self.failures = tuple(failures)


def parse_smiles(smiles: Sequence[str]) -> list[Mol]:
    """Parse every SMILES string, raising if any of them fails.

    Note that a SMILES may describe several disconnected components separated by
    ``.`` -- salts such as ``[Cl-].[K+]``. Those parse fine and are kept as one
    entry; whether they belong in an odour dataset at all is a data decision, not
    a parsing one.
    """
    molecules: list[Mol] = []
    failures: list[tuple[int, str]] = []
    for index, text in enumerate(smiles):
        molecule = Chem.MolFromSmiles(text)
        if molecule is None:
            failures.append((index, text))
        else:
            molecules.append(molecule)
    if failures:
        raise InvalidSmilesError(failures)
    return molecules
