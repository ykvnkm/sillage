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


# The two expert collections OpenPOM merged, before the merge: same molecules, labels
# kept apart. Used to measure how far the experts agree with each other -- the
# ceiling any model of their labels can reach (journal E-018) -- and later to recover
# the stereochemistry the merge dropped (E-015).
#
# Pyrfume ships no licence file for either. Goodscents is a scrape of a commercial
# website, Leffingwell's terms sit in a PDF under "use restrictions". Treated the way
# CONCEPT treats scraped data: download and measure locally, publish only derived
# aggregates, never the records.
_PYRFUME_COMMIT = "8054ea98ed675005ec10e67359902f500e4911b0"
_PYRFUME_BASE = f"https://raw.githubusercontent.com/pyrfume/pyrfume-data/{_PYRFUME_COMMIT}"


def _pyrfume(collection: str, table: str, sha256: str, description: str) -> DataSource:
    return DataSource(
        name=f"pyrfume-{collection}-{table}",
        url=f"{_PYRFUME_BASE}/{collection}/{table}.csv",
        filename=f"pyrfume_{collection}_{table}.csv",
        sha256=sha256,
        licence="no licence file; source terms apply",
        homepage=f"https://github.com/pyrfume/pyrfume-data/tree/{_PYRFUME_COMMIT}/{collection}",
        redistribution=Redistribution.DERIVED_ONLY,
        description=description,
    )


PYRFUME_SOURCES = (
    _pyrfume(
        "leffingwell",
        "molecules",
        "a06841a275c35b9db9253a24ac913ad97bf39f2c778e4f8a5c4df641e7dc0fa6",
        "Leffingwell: structures by stimulus id.",
    ),
    _pyrfume(
        "leffingwell",
        "behavior",
        "afe0846b159ac00691ba4c07074829f7ff262b80fa32a3280661457153e70c41",
        "Leffingwell: 113 binary odour labels.",
    ),
    _pyrfume(
        "leffingwell",
        "stimuli",
        "ff4be9ba6a66523bc5a165ded09eee125ccb79d4988090d682b323d9331a44c8",
        "Leffingwell: stimulus to PubChem id.",
    ),
    _pyrfume(
        "goodscents",
        "molecules",
        "0d9bd38c54a40a051ae7b771d51f11ac1fcf021e0369fc53583e5f634e6e8bb4",
        "Goodscents: isomeric structures by PubChem id.",
    ),
    _pyrfume(
        "goodscents",
        "behavior",
        "18968aa66cab974ee2f942ad0abd0875ccf2bcc9501c1cb306a54bd639a22219",
        "Goodscents: free-text odour descriptors.",
    ),
    _pyrfume(
        "goodscents",
        "stimuli",
        "356cdd74a859ab1e8e061952dbb21587afc8981bb460cc670481d688afac923f",
        "Goodscents: record to PubChem id.",
    ),
)

REGISTRY: dict[str, DataSource] = {
    source.name: source for source in (OPENPOM_CURATED, *PYRFUME_SOURCES)
}


def get(name: str) -> DataSource:
    """Look up a source by name, failing with the list of valid names."""
    try:
        return REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(REGISTRY)) or "<empty registry>"
        raise KeyError(f"Unknown source {name!r}. Known sources: {known}") from None
