"""Splitting the dataset into train, validation and test.

Two strategies, kept side by side on purpose. The stratified split measures
performance on molecules resembling the training set; the scaffold split measures
performance on structures the model has never seen. Reporting only the first is
optimistic, reporting only the second hides how the rare labels behave.
"""

from sillage.splits.scaffold import group_split, scaffold_keys, scaffold_split
from sillage.splits.stratified import iterative_stratification
from sillage.splits.types import PARTS, Split, label_coverage

__all__ = [
    "PARTS",
    "Split",
    "group_split",
    "iterative_stratification",
    "label_coverage",
    "scaffold_keys",
    "scaffold_split",
]
