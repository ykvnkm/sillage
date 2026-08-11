"""The contract must accept valid data and reject each way it can go wrong.

A schema that only ever sees good input is untested: it would pass just as happily
if half its checks were deleted. Every test below breaks exactly one assumption.
"""

from __future__ import annotations

import pandas as pd
import pytest
from pandera.errors import SchemaError, SchemaErrors

from sillage.data.schemas import (
    DESCRIPTORS_COLUMN,
    OPENPOM_DESCRIPTORS,
    SMILES_COLUMN,
    curated_openpom_schema,
)


@pytest.fixture
def valid_frame() -> pd.DataFrame:
    """Three molecules shaped exactly like the real file, in miniature."""
    rows = [
        ("CCO", ("alcoholic",)),
        ("CC(O)CN", ("fishy",)),
        ("O=C(O)CCc1ccccc1", ("rose", "floral", "sweet")),
    ]
    # Built in one shot rather than column by column: 138 successive inserts
    # fragment the frame and pandas rightly complains about it.
    return pd.DataFrame(
        {
            SMILES_COLUMN: [smiles for smiles, _ in rows],
            DESCRIPTORS_COLUMN: [";".join(labels) for _, labels in rows],
            **{name: [int(name in labels) for _, labels in rows] for name in OPENPOM_DESCRIPTORS},
        }
    )


def validate(frame: pd.DataFrame) -> pd.DataFrame:
    return curated_openpom_schema().validate(frame)


def test_valid_frame_passes(valid_frame: pd.DataFrame) -> None:
    assert validate(valid_frame).shape == valid_frame.shape


def test_non_binary_label_is_rejected(valid_frame: pd.DataFrame) -> None:
    """Counts instead of indicators would break every metric silently."""
    valid_frame.loc[0, "alcoholic"] = 2

    with pytest.raises((SchemaError, SchemaErrors)):
        validate(valid_frame)


def test_duplicate_molecule_is_rejected(valid_frame: pd.DataFrame) -> None:
    """A molecule present twice can land in train and test at once: leakage."""
    valid_frame.loc[1, SMILES_COLUMN] = valid_frame.loc[0, SMILES_COLUMN]

    with pytest.raises((SchemaError, SchemaErrors)):
        validate(valid_frame)


def test_row_without_any_label_is_rejected(valid_frame: pd.DataFrame) -> None:
    valid_frame.loc[0, list(OPENPOM_DESCRIPTORS)] = 0
    valid_frame.loc[0, DESCRIPTORS_COLUMN] = ""

    with pytest.raises((SchemaError, SchemaErrors)):
        validate(valid_frame)


def test_descriptor_text_disagreeing_with_columns_is_rejected(valid_frame: pd.DataFrame) -> None:
    """The redundancy catches column shifts that parsing alone would not notice."""
    valid_frame.loc[0, DESCRIPTORS_COLUMN] = "woody"

    with pytest.raises((SchemaError, SchemaErrors)):
        validate(valid_frame)


def test_unknown_column_is_rejected(valid_frame: pd.DataFrame) -> None:
    """An added descriptor changes what "all labels" means for every downstream loop."""
    valid_frame["brand new descriptor"] = 0

    with pytest.raises((SchemaError, SchemaErrors)):
        validate(valid_frame)


def test_missing_descriptor_column_is_rejected(valid_frame: pd.DataFrame) -> None:
    with pytest.raises((SchemaError, SchemaErrors)):
        validate(valid_frame.drop(columns=["ozone"]))


def test_empty_smiles_is_rejected(valid_frame: pd.DataFrame) -> None:
    valid_frame.loc[0, SMILES_COLUMN] = ""

    with pytest.raises((SchemaError, SchemaErrors)):
        validate(valid_frame)


def test_descriptor_set_has_no_duplicates() -> None:
    assert len(set(OPENPOM_DESCRIPTORS)) == len(OPENPOM_DESCRIPTORS) == 138
