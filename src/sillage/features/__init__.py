"""Turning molecules into numbers.

Every featurizer here is a **pure function of a single molecule**: the vector for
a molecule does not depend on which other molecules were passed alongside it, nor
on any statistic of the dataset. That property is what makes it safe to featurize
before splitting the data.

Anything that *learns* from the data -- scaling, feature selection, imputation by
column mean -- must instead be fitted on the training split only, and therefore
lives in the model pipeline rather than here.
"""

from sillage.features.descriptors import (
    DESCRIPTOR_NAMES,
    EXCLUDED_DESCRIPTORS,
    rdkit_descriptors,
)
from sillage.features.fingerprints import morgan_fingerprints
from sillage.features.matrix import FeatureMatrix
from sillage.features.molecules import InvalidSmilesError, parse_smiles

__all__ = [
    "DESCRIPTOR_NAMES",
    "EXCLUDED_DESCRIPTORS",
    "FeatureMatrix",
    "InvalidSmilesError",
    "morgan_fingerprints",
    "parse_smiles",
    "rdkit_descriptors",
]
