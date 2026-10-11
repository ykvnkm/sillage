"""What one prediction request ends in: an answer, or a refusal with a reason."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from sillage.domain.molecule import KnownMolecule, Molecule, Neighbour
from sillage.domain.odor import OdorProfile


class RefusalReason(StrEnum):
    """Why the service declines to predict. Stable codes: the API and the metrics
    exported for drift monitoring both key on these values."""

    INVALID_INPUT = "invalid_input"
    MULTIPLE_COMPONENTS = "multiple_components"
    INORGANIC = "inorganic"
    CHARGED = "charged"
    UNSEEN_ELEMENT = "unseen_element"
    WEIGHT_OUT_OF_RANGE = "weight_out_of_range"


@dataclass(frozen=True, slots=True)
class Refusal:
    """A declined request. Not an error: refusing is a designed answer.

    Outside the declared scope the model is about six times more likely to fail
    catastrophically, and on an unseen element it makes seventeen times more
    confident claims (journal E-008, E-010). Saying "this is outside what I was
    built for" is the correct output for those inputs.
    """

    reason: RefusalReason
    detail: str
    molecule: Molecule | None = None
    """Present when the input parsed but fell outside the scope."""


class Caveat(StrEnum):
    """Something true about this particular answer that the user must be told."""

    STEREO_DISCARDED = "stereo_discarded"
    """The input specified stereochemistry; the model cannot see it."""


@dataclass(frozen=True, slots=True)
class OdorReport:
    """A successful prediction, with everything needed to read it honestly."""

    molecule: Molecule
    profile: OdorProfile
    neighbours: tuple[Neighbour, ...]
    known: KnownMolecule | None
    """What the dataset says when this exact molecule is in it, otherwise None.

    Shown next to the prediction, never instead of it. For a known molecule the
    useful answer is both: what experts recorded, and whether the model agrees."""

    caveats: tuple[Caveat, ...]
    model_name: str
