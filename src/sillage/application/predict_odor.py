"""SMILES in, an honest odour report out.

The order of the steps is the design:

1. **Read.** Text that is not a molecule is refused before anything else runs.
2. **Check scope.** A salt, an inorganic or a bromine compound is refused *before*
   the model is called, so no number about it ever exists to leak into a response.
3. **Predict**, and attach the context that makes the prediction readable: which
   known molecules it resembles, what experts said if this one is already known,
   and what the model could not see.
"""

from __future__ import annotations

from dataclasses import dataclass

from sillage.domain import (
    Caveat,
    InvalidMolecule,
    OdorReport,
    Refusal,
    RefusalReason,
)
from sillage.domain.ports import MoleculeReader, OdorPredictor, ReferenceLibrary, ScopePolicy

MAX_INPUT_LENGTH = 500
"""Characters. Everything the model is defined for (up to 400 Da) fits in about a
hundred; the limit exists so that a public endpoint cannot be handed a megabyte of
text to parse."""

DEFAULT_NEIGHBOURS = 5


@dataclass(frozen=True, slots=True)
class PredictOdor:
    reader: MoleculeReader
    scope: ScopePolicy
    predictor: OdorPredictor
    references: ReferenceLibrary

    def __call__(self, text: str, *, neighbours: int = DEFAULT_NEIGHBOURS) -> OdorReport | Refusal:
        text = text.strip()
        if not text:
            return Refusal(RefusalReason.INVALID_INPUT, "empty input")
        if len(text) > MAX_INPUT_LENGTH:
            return Refusal(
                RefusalReason.INVALID_INPUT,
                f"input is {len(text)} characters long, the limit is {MAX_INPUT_LENGTH}",
            )

        try:
            molecule = self.reader.read(text)
        except InvalidMolecule as error:
            return Refusal(RefusalReason.INVALID_INPUT, str(error))

        refusal = self.scope.check(molecule)
        if refusal is not None:
            return refusal

        caveats = (Caveat.STEREO_DISCARDED,) if molecule.stereo_discarded else ()
        return OdorReport(
            molecule=molecule,
            profile=self.predictor.predict(molecule),
            neighbours=self.references.nearest(molecule, neighbours),
            known=self.references.lookup(molecule),
            caveats=caveats,
            model_name=self.predictor.name,
        )
