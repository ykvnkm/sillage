"""RDKit physicochemical descriptors.

Where a fingerprint answers "which fragments are present", descriptors answer
"what is this molecule like": how heavy, how greasy, how many rings, how many
hydrogen-bond donors. There are a couple of hundred of them, they are
interpretable, and they are exactly the numbers a perfumer reasons about when
guessing whether something is a top note or a base note.

Two RDKit quirks are handled here, both discovered on this dataset rather than
read about:

*Ipc overflows.* The information-content index grows exponentially with molecular
size. On the cyclodextrins in this dataset it reaches 1.5e24 against a median of
489. A single feature spanning twenty orders of magnitude wrecks any linear model
and is worthless to a tree, so it is excluded outright.

*Partial charges are undefined for salts.* Every Gasteiger-derived descriptor
(``MaxPartialCharge`` and the ``BCUT2D_*`` family) returns NaN for the 108 ionic
entries such as ``[Cl-].[K+]``. Non-finite values are replaced with 0.0 rather
than dropping either the column or the row: the column is informative for the
other 98% of molecules, and dropping rows would silently shrink the dataset.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from rdkit.Chem import Descriptors, Mol

from sillage.features.matrix import FeatureMatrix

EXCLUDED_DESCRIPTORS: frozenset[str] = frozenset({"Ipc"})

# Order comes from RDKit and is stable within a version. It is *not* stable across
# versions -- an upgrade may add descriptors and silently change the feature space.
# tests/test_features.py pins the count so that such an upgrade fails loudly.
DESCRIPTOR_NAMES: tuple[str, ...] = tuple(
    name for name, _ in Descriptors._descList if name not in EXCLUDED_DESCRIPTORS
)

_CALCULATORS = tuple(
    function for name, function in Descriptors._descList if name not in EXCLUDED_DESCRIPTORS
)


def rdkit_descriptors(molecules: Sequence[Mol]) -> FeatureMatrix:
    """Compute the physicochemical descriptor block.

    Non-finite results (NaN from undefined partial charges, any infinity) are
    replaced with 0.0. Constant and near-constant columns are *not* removed here:
    that is feature selection, it depends on the data, and doing it before the
    split would leak information from the test set.
    """
    if not molecules:
        values = np.zeros((0, len(DESCRIPTOR_NAMES)), dtype=np.float64)
    else:
        values = np.array(
            [[calculate(molecule) for calculate in _CALCULATORS] for molecule in molecules],
            dtype=np.float64,
        )

    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return FeatureMatrix(values=values.astype(np.float32), names=DESCRIPTOR_NAMES)
