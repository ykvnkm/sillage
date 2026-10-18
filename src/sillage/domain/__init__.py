"""The product's vocabulary: what a molecule, an odour profile and a refusal *are*.

Standard library only. Nothing here may import RDKit, NumPy, scikit-learn or any
other third-party package, and ``tests/test_layers.py`` fails the build if it does
(ADR-0001). The reason is not purity for its own sake: these types are what the
API returns and what every future model -- the GNN of phase 2, the mixture model of
phase 4 -- has to produce. They must not change because a library did.
"""

from sillage.domain.families import FAMILY_NAMES_RU, FAMILY_OF, Family
from sillage.domain.molecule import InvalidMolecule, KnownMolecule, Molecule, Neighbour
from sillage.domain.names import MoleculeNames, NameSuggestion, fold, initial, looks_like_name
from sillage.domain.odor import (
    BANDS,
    UNVERIFIED_MENTION,
    Confidence,
    DescriptorScore,
    ModelDescription,
    NoteExplanation,
    OdorProfile,
    confidence_of,
)
from sillage.domain.odor_map import MapPoint, MapPosition
from sillage.domain.report import Caveat, OdorReport, Refusal, RefusalReason

__all__ = [
    "BANDS",
    "FAMILY_NAMES_RU",
    "FAMILY_OF",
    "UNVERIFIED_MENTION",
    "Caveat",
    "Confidence",
    "DescriptorScore",
    "Family",
    "InvalidMolecule",
    "KnownMolecule",
    "MapPoint",
    "MapPosition",
    "ModelDescription",
    "Molecule",
    "MoleculeNames",
    "NameSuggestion",
    "Neighbour",
    "NoteExplanation",
    "OdorProfile",
    "OdorReport",
    "Refusal",
    "RefusalReason",
    "confidence_of",
    "fold",
    "initial",
    "looks_like_name",
]
