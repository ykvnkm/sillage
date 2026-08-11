"""Reading raw files into validated DataFrames.

Loading and validating are deliberately one operation. If they were separate,
validation would eventually be skipped "just this once" in a notebook, and the
guarantee would quietly stop holding where it matters most.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sillage.data.schemas import (
    DESCRIPTORS_COLUMN,
    OPENPOM_DESCRIPTORS,
    SMILES_COLUMN,
    curated_openpom_schema,
)
from sillage.data.sources import OPENPOM_CURATED
from sillage.paths import raw_data_dir


def load_curated_openpom(path: Path | None = None, *, validate: bool = True) -> pd.DataFrame:
    """Load the curated Goodscents/Leffingwell merge.

    Args:
        path: file to read; defaults to the download location in ``data/raw``.
        validate: run the contract. Only turn this off to inspect a file that is
            already known to be broken.

    Raises:
        FileNotFoundError: when the dataset has not been downloaded yet.
        pandera.errors.SchemaError: when the file violates the contract.
    """
    target = path if path is not None else raw_data_dir() / OPENPOM_CURATED.filename
    if not target.exists():
        raise FileNotFoundError(
            f"{target} is missing. Download it first:\n"
            f"    uv run sillage-data fetch {OPENPOM_CURATED.name}"
        )

    frame = pd.read_csv(
        target,
        dtype={SMILES_COLUMN: str, DESCRIPTORS_COLUMN: str}
        | dict.fromkeys(OPENPOM_DESCRIPTORS, "int64"),
    )
    return curated_openpom_schema().validate(frame) if validate else frame
