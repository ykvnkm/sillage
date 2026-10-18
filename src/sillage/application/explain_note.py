"""Why does the model hear this note? The atoms that carry it (journal E-021).

The same gates as a prediction, in the same order: text that is not a molecule, and a
molecule outside the model's scope, are refused before the model is asked anything.
The input is a SMILES string -- the one the prediction report returned -- because an
explanation is always asked about a molecule the user is already looking at.
"""

from __future__ import annotations

from dataclasses import dataclass

from sillage.application.predict_odor import MAX_INPUT_LENGTH
from sillage.domain import InvalidMolecule, NoteExplanation, Refusal, RefusalReason
from sillage.domain.ports import MoleculeReader, OdorExplainer, ScopePolicy


@dataclass(frozen=True, slots=True)
class ExplainNote:
    reader: MoleculeReader
    scope: ScopePolicy
    explainer: OdorExplainer

    def __call__(self, smiles: str, descriptor: str) -> NoteExplanation | Refusal:
        text = smiles.strip()
        if not text or len(text) > MAX_INPUT_LENGTH:
            return Refusal(RefusalReason.INVALID_INPUT, "expected a SMILES string")
        try:
            molecule = self.reader.read(text)
        except InvalidMolecule as error:
            return Refusal(RefusalReason.INVALID_INPUT, str(error))

        refusal = self.scope.check(molecule)
        if refusal is not None:
            return refusal
        try:
            return self.explainer.explain(molecule, descriptor)
        except KeyError:
            return Refusal(
                RefusalReason.INVALID_INPUT, f"the model has no descriptor {descriptor!r}"
            )
