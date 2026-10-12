"""A name or a SMILES string in, an honest odour report out.

The order of the steps is the design:

1. **Resolve the input.** A known name from the local directory first (instant, no
   network), then SMILES, then an online lookup for names the directory lacks. Text
   that is neither is refused before anything else runs.
2. **Check scope.** A salt, an inorganic or a bromine compound is refused *before*
   the model is called, so no number about it ever exists to leak into a response.
3. **Predict**, and attach the context that makes the prediction readable: what the
   molecule is called, which known molecules it resembles, what experts said if this
   one is already known, and what the model could not see.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from sillage.domain import (
    Caveat,
    InvalidMolecule,
    Molecule,
    MoleculeNames,
    OdorReport,
    Refusal,
    RefusalReason,
    looks_like_name,
)
from sillage.domain.ports import (
    MoleculeReader,
    NameDirectory,
    NameLookupUnavailable,
    NameResolver,
    OdorPredictor,
    ReferenceLibrary,
    ScopePolicy,
)

MAX_INPUT_LENGTH = 500
"""Characters. Everything the model is defined for (up to 400 Da) fits in about a
hundred; the limit exists so that a public endpoint cannot be handed a megabyte of
text to parse."""

DEFAULT_NEIGHBOURS = 5


@dataclass(frozen=True, slots=True)
class _Resolved:
    molecule: Molecule
    names: MoleculeNames | None
    kind: str
    matched: str | None = None


@dataclass(frozen=True, slots=True)
class PredictOdor:
    reader: MoleculeReader
    scope: ScopePolicy
    predictor: OdorPredictor
    references: ReferenceLibrary
    names: NameDirectory | None = None
    resolver: NameResolver | None = None

    def __call__(self, text: str, *, neighbours: int = DEFAULT_NEIGHBOURS) -> OdorReport | Refusal:
        text = text.strip()
        if not text:
            return Refusal(RefusalReason.INVALID_INPUT, "empty input")
        if len(text) > MAX_INPUT_LENGTH:
            return Refusal(
                RefusalReason.INVALID_INPUT,
                f"input is {len(text)} characters long, the limit is {MAX_INPUT_LENGTH}",
            )

        resolved = self._resolve(text)
        if isinstance(resolved, Refusal):
            return resolved
        molecule = resolved.molecule

        refusal = self.scope.check(molecule)
        if refusal is not None:
            return refusal

        caveats = (Caveat.STEREO_DISCARDED,) if molecule.stereo_discarded else ()
        return OdorReport(
            molecule=molecule,
            profile=self.predictor.predict(molecule),
            neighbours=tuple(
                replace(n, names=self._names_of(n.smiles))
                for n in self.references.nearest(molecule, neighbours)
            ),
            known=self.references.lookup(molecule),
            caveats=caveats,
            model_name=self.predictor.name,
            names=resolved.names,
            query_kind=resolved.kind,
            matched_name=resolved.matched,
        )

    def _resolve(self, text: str) -> _Resolved | Refusal:
        if self.names is not None and (found := self.names.find(text)) is not None:
            smiles, names, matched = found
            return _Resolved(self.reader.read(smiles), names, "name", matched)

        try:
            molecule = self.reader.read(text)
        except InvalidMolecule as error:
            if not looks_like_name(text):
                return Refusal(RefusalReason.INVALID_INPUT, str(error))
        else:
            return _Resolved(molecule, self._names_of(molecule.smiles), "smiles")

        if self.resolver is None:
            return Refusal(RefusalReason.UNKNOWN_NAME, f"no molecule called {text!r} is known")
        try:
            found = self.resolver.resolve(text)
        except NameLookupUnavailable as error:
            return Refusal(RefusalReason.NAME_LOOKUP_UNAVAILABLE, str(error))
        if found is None:
            return Refusal(RefusalReason.UNKNOWN_NAME, f"no molecule called {text!r} is known")

        smiles, names = found
        try:
            molecule = self.reader.read(smiles)
        except InvalidMolecule as error:
            return Refusal(RefusalReason.UNKNOWN_NAME, str(error))
        # The local directory knows the Russian name; the online service does not.
        return _Resolved(molecule, self._names_of(molecule.smiles) or names, "name", names.en)

    def _names_of(self, smiles: str) -> MoleculeNames | None:
        return None if self.names is None else self.names.names_of(smiles)
