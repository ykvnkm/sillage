"""Data contracts for the raw datasets.

A checksum proves the bytes are the ones we expect. It says nothing about whether
those bytes still *mean* what the code assumes. A contract states the assumptions
explicitly and checks them on load, so that a violation surfaces as a loud failure
at the boundary rather than as an inexplicable metric twenty minutes into training.

Every assumption encoded here was verified against the pinned revision of the
dataset before being written down; none of it is aspirational.
"""

from __future__ import annotations

import pandas as pd
from pandera.pandas import Check, Column, DataFrameSchema

SMILES_COLUMN = "nonStereoSMILES"
DESCRIPTORS_COLUMN = "descriptors"
DESCRIPTOR_SEPARATOR = ";"

# The 138 odour descriptors of the curated Goodscents/Leffingwell merge, in file
# order. Listing them is the contract: if upstream adds, removes or renames one,
# loading must fail rather than silently changing the shape of the task.
OPENPOM_DESCRIPTORS: tuple[str, ...] = (
    "alcoholic", "aldehydic", "alliaceous", "almond", "amber", "animal", "anisic", "apple",
    "apricot", "aromatic", "balsamic", "banana", "beefy", "bergamot", "berry", "bitter",
    "black currant", "brandy", "burnt", "buttery", "cabbage", "camphoreous", "caramellic",
    "cedar", "celery", "chamomile", "cheesy", "cherry", "chocolate", "cinnamon", "citrus",
    "clean", "clove", "cocoa", "coconut", "coffee", "cognac", "cooked", "cooling", "cortex",
    "coumarinic", "creamy", "cucumber", "dairy", "dry", "earthy", "ethereal", "fatty",
    "fermented", "fishy", "floral", "fresh", "fruit skin", "fruity", "garlic", "gassy",
    "geranium", "grape", "grapefruit", "grassy", "green", "hawthorn", "hay", "hazelnut",
    "herbal", "honey", "hyacinth", "jasmin", "juicy", "ketonic", "lactonic", "lavender",
    "leafy", "leathery", "lemon", "lily", "malty", "meaty", "medicinal", "melon", "metallic",
    "milky", "mint", "muguet", "mushroom", "musk", "musty", "natural", "nutty", "odorless",
    "oily", "onion", "orange", "orangeflower", "orris", "ozone", "peach", "pear", "phenolic",
    "pine", "pineapple", "plum", "popcorn", "potato", "powdery", "pungent", "radish",
    "raspberry", "ripe", "roasted", "rose", "rummy", "sandalwood", "savory", "sharp", "smoky",
    "soapy", "solvent", "sour", "spicy", "strawberry", "sulfurous", "sweaty", "sweet", "tea",
    "terpenic", "tobacco", "tomato", "tropical", "vanilla", "vegetable", "vetiver", "violet",
    "warm", "waxy", "weedy", "winey", "woody",
)  # fmt: skip

EXPECTED_ROW_COUNT = 4983


def _listed_descriptors(cell: str) -> frozenset[str]:
    return frozenset(part.strip() for part in cell.split(DESCRIPTOR_SEPARATOR) if part.strip())


def _encoded_descriptors(row: pd.Series) -> frozenset[str]:
    return frozenset(name for name in OPENPOM_DESCRIPTORS if row[name] == 1)


def _has_at_least_one_label(frame: pd.DataFrame) -> pd.Series:
    """A molecule with no descriptors carries no signal and no supervision."""
    return frame.loc[:, list(OPENPOM_DESCRIPTORS)].sum(axis=1) > 0


def _descriptor_text_matches_one_hot(frame: pd.DataFrame) -> pd.Series:
    """The redundancy in the file is the point: disagreement means corruption.

    Every row stores its labels twice -- once as a ``;``-joined string, once as 138
    binary columns. On the pinned revision they agree everywhere. A column shift or
    a partially rewritten file breaks that agreement long before it breaks parsing.
    """
    return frame.apply(
        lambda row: _listed_descriptors(row[DESCRIPTORS_COLUMN]) == _encoded_descriptors(row),
        axis=1,
    )


def curated_openpom_schema() -> DataFrameSchema:
    """Contract for ``curated_GS_LF_merged_4983.csv``."""
    label_columns = {
        name: Column(
            int,
            checks=Check.isin((0, 1), error="label is not binary"),
            nullable=False,
            required=True,
        )
        for name in OPENPOM_DESCRIPTORS
    }
    return DataFrameSchema(
        columns={
            SMILES_COLUMN: Column(
                str,
                nullable=False,
                unique=True,
                checks=Check.str_length(min_value=1, error="empty SMILES"),
            ),
            DESCRIPTORS_COLUMN: Column(str, nullable=False),
            **label_columns,
        },
        checks=[
            Check(_has_at_least_one_label, error="row has no positive label"),
            Check(
                _descriptor_text_matches_one_hot,
                error="descriptors text disagrees with the one-hot columns",
            ),
        ],
        # Reject unknown columns: an added column silently changes what "all labels"
        # means for every downstream loop over the descriptor set.
        strict=True,
        coerce=False,
        name="curated_GS_LF_merged_4983",
    )


__all__ = [
    "DESCRIPTORS_COLUMN",
    "EXPECTED_ROW_COUNT",
    "OPENPOM_DESCRIPTORS",
    "SMILES_COLUMN",
    "curated_openpom_schema",
]
