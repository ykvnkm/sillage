"""Declarative registry of external data sources.

Each source is described once and that description serves three purposes:
downloading it, verifying it did not change, and generating the "Data provenance"
section of the README. The raw bytes themselves never enter the repository --
only these descriptions and the checksums that identify them.

Adding a source means adding an entry here, not writing another script.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Redistribution(StrEnum):
    """What this project is allowed to republish, as opposed to merely download."""

    ALLOWED = "allowed"
    """Permissive licence: the data itself may be redistributed with attribution."""

    DERIVED_ONLY = "derived-only"
    """Only derived artefacts may be published: weights, embeddings, aggregates."""

    FORBIDDEN = "forbidden"
    """Neither the data nor close derivatives may be republished."""


@dataclass(frozen=True, slots=True)
class DataSource:
    """One downloadable artefact, pinned to an exact revision and checksum."""

    name: str
    """Stable identifier. Used as the CLI argument; never rename it casually."""

    url: str
    """Direct link. Must address an immutable revision (a commit, not a branch)."""

    filename: str
    """Name the file is stored under in data/raw/."""

    sha256: str | None
    """Expected digest, or None until the first download pins it (trust on first use)."""

    licence: str
    homepage: str
    redistribution: Redistribution
    description: str

    @property
    def is_pinned(self) -> bool:
        """False while the checksum is still unknown, i.e. integrity is unverified."""
        return self.sha256 is not None


# The curated Goodscents + Leffingwell merge shipped inside OpenPOM: 4983 molecules
# labelled with 138 odour descriptors. Primary training set for L0.
# Pinned to the commit that last touched the file (2023-08-20), not to a branch:
# a branch URL silently changes contents and turns integrity checks into noise.
_OPENPOM_COMMIT = "8c345c1451bc3660c1e38575814d793beff2ed9f"

OPENPOM_CURATED = DataSource(
    name="openpom-curated-gs-lf",
    url=(
        f"https://raw.githubusercontent.com/BioMachineLearning/openpom/{_OPENPOM_COMMIT}"
        "/openpom/data/curated_datasets/curated_GS_LF_merged_4983.csv"
    ),
    filename="curated_GS_LF_merged_4983.csv",
    sha256="0c18b6e9f0f99b772203ba4da30a96b3d60db700349376cbafe23512a9baa704",
    licence="MIT",
    homepage="https://github.com/BioMachineLearning/openpom",
    redistribution=Redistribution.ALLOWED,
    description="Curated Goodscents/Leffingwell merge: 4983 molecules, multilabel odours.",
)


REGISTRY: dict[str, DataSource] = {source.name: source for source in (OPENPOM_CURATED,)}


def get(name: str) -> DataSource:
    """Look up a source by name, failing with the list of valid names."""
    try:
        return REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(REGISTRY)) or "<empty registry>"
        raise KeyError(f"Unknown source {name!r}. Known sources: {known}") from None
