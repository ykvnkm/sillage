"""Ports: what the application needs from the outside world, stated as protocols.

The application layer calls these and nothing else. Which library sits behind each
one -- RDKit, scikit-learn, later PyTorch Geometric -- is decided in
``sillage.interfaces.composition`` and nowhere above it.

``OdorPredictor`` is the port ADR-0001 justified in advance: the phase 1 baseline and
the phase 2 GNN are two implementations of it, compared on the same split, and the
API must not notice which one is plugged in.
"""

from __future__ import annotations

from typing import Protocol

from sillage.domain.molecule import KnownMolecule, Molecule, Neighbour
from sillage.domain.names import MoleculeNames, NameSuggestion
from sillage.domain.odor import ModelDescription, NoteExplanation, OdorProfile
from sillage.domain.odor_map import MapPoint, MapPosition
from sillage.domain.report import Refusal


class MoleculeReader(Protocol):
    """Turns user text into a :class:`Molecule`."""

    def read(self, text: str) -> Molecule:
        """Raises :class:`~sillage.domain.molecule.InvalidMolecule` on bad input."""
        ...


class ScopePolicy(Protocol):
    """Decides whether the model may be asked about a molecule at all."""

    def check(self, molecule: Molecule) -> Refusal | None: ...


class OdorPredictor(Protocol):
    """Structure -> odour. One implementation per model family."""

    @property
    def name(self) -> str: ...

    def predict(self, molecule: Molecule) -> OdorProfile: ...

    def embed(self, molecule: Molecule) -> tuple[float, ...]:
        """The model's internal representation of the molecule's smell.

        For the baseline: its 138 predicted descriptor probabilities. For a neural
        model: its penultimate layer. Whatever it is, the odour map is built from it.
        """
        ...

    def describe(self) -> ModelDescription: ...

    @property
    def sees_stereo(self) -> bool:
        """Whether two stereoisomers can get different answers from this model."""
        ...


class OdorExplainer(Protocol):
    """Which atoms carry a note. Not every model can say: one without fingerprints or
    graphs to trace has nothing to point at, and is simply not wired as one."""

    def explain(self, molecule: Molecule, descriptor: str) -> NoteExplanation:
        """Raises :class:`KeyError` for a descriptor the model does not know."""
        ...


class OdorMap(Protocol):
    """Molecules laid out by how the model thinks they smell."""

    def locate(self, embedding: tuple[float, ...]) -> MapPosition: ...

    def points(self) -> tuple[MapPoint, ...]: ...


class NameDirectory(Protocol):
    """Names of known molecules, searchable offline and instantly."""

    def find(self, text: str) -> tuple[str, MoleculeNames, str] | None:
        """``(smiles, names, matched)`` for a name that matches exactly, after folding.

        *matched* is the name as the directory spells it -- "Iso-E super", not the
        70-character IUPAC title -- so the page can lead with the name the user knows.
        """
        ...

    def names_of(self, smiles: str) -> MoleculeNames | None:
        """Names for a canonical, stereo-free SMILES, if the molecule is known."""
        ...

    def suggest(self, text: str, limit: int) -> tuple[NameSuggestion, ...]:
        """Names that start like *text*, best first. For autocomplete."""
        ...


class NameLookupUnavailable(RuntimeError):
    """The online name service could not be reached or answered with an error."""


class NameResolver(Protocol):
    """Resolves names the local directory does not know, through an online service."""

    def resolve(self, name: str) -> tuple[str, MoleculeNames] | None:
        """``(smiles, names)``, None when the service has no such name.

        Raises :class:`NameLookupUnavailable` when the service cannot answer.
        """
        ...


class MoleculeDepicter(Protocol):
    """Draws a molecule for a person to look at."""

    def svg(
        self, smiles: str, width: int, height: int, weights: tuple[float, ...] | None = None
    ) -> str:
        """Raises :class:`~sillage.domain.molecule.InvalidMolecule` on bad input.

        *weights*, one per atom of *smiles* in its own order, shade the atoms: what
        carries a note in one colour, what holds it back in another."""
        ...


class ReferenceLibrary(Protocol):
    """Molecules with known expert labels, searchable by structure."""

    def lookup(self, molecule: Molecule) -> KnownMolecule | None:
        """The dataset record for this exact molecule, or None if it is not in the library."""
        ...

    def nearest(self, molecule: Molecule, k: int) -> tuple[Neighbour, ...]:
        """The *k* most similar library molecules, most similar first, itself excluded."""
        ...
