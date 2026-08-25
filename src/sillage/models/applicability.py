"""Applicability domain: deciding when the model has no business answering.

Error analysis in step 1.6 found the failure that matters most for a public
service. Basic magnesium carbonate -- an inorganic salt -- receives a score of
1.000 for seven odour descriptors and 0.000 for ``odorless``, its only true label.
It lights 8 Morgan bits against a dataset average of 29: its feature vector sits
outside anything the model has seen, and a linear model extrapolates into that void
with complete confidence.

The model cannot say "I do not know". A score of 1.000 means both "confidently
fishy" and "no idea what this is", and nothing in the output distinguishes them.

This module holds two answers to that, and **only the second one works**. The first
is kept because a refuted idea is only credible while it stays reproducible.

:class:`ApplicabilityDomain` is the textbook approach: refuse molecules whose
nearest training neighbour is too dissimilar, with the threshold calibrated on a
held-out split. Measured on this dataset it is **indistinguishable from refusing at
random** -- at 95% coverage both give mean per-molecule AP 0.571. It does not even
flag the magnesium carbonate, whose Tanimoto similarity is 0.875: the measure is a
ratio, so a molecule with 8 bits sharing 7 of them scores high. The anomaly was
never "no similar molecule exists", it was "there is almost nothing here to compare".
Journal entry E-008 has the full curve.

:class:`ChemicalScope` is the answer that holds. It does not learn a threshold; it
states what the model is for. Magnesium carbonate is not a hard case for an odour
model -- it is an inorganic salt, outside the question being asked. Out-of-scope
molecules turn out to be about six times more likely to fail catastrophically.

The similarity code remains useful in its own right: "the closest molecule I know is
X, 0.87 similar" is worth showing a user even when it cannot gate a prediction.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Mol

DEFAULT_COVERAGE = 0.95


@dataclass(frozen=True, slots=True)
class ChemicalScope:
    """What the model is *defined* for, stated rather than learned.

    Measured on this dataset, :class:`ApplicabilityDomain` -- a learned threshold on
    nearest-neighbour similarity -- performs no better than refusing at random
    (journal E-008). A declared scope does something different and more useful: it
    refuses inputs the model was never built for, by definition instead of by
    threshold.

    Basic magnesium carbonate is not a hard case for an odour model. It is an
    inorganic salt, outside the question the model answers. Saying so in one rule is
    honest, explainable to a user, and cannot drift.

    Measured: out-of-scope molecules are about six times more likely to be a
    catastrophic failure (20% against 3.4%), while making up 2.9% of the test set.
    """

    min_weight: float = 30.0
    max_weight: float = 400.0
    """Above roughly 400 Da a molecule is too heavy to evaporate and reach the
    olfactory epithelium; below 30 there is hardly a molecule."""

    require_carbon: bool = True
    require_single_component: bool = True
    require_neutral: bool = True

    def reason_for_rejection(self, molecule: Mol) -> str | None:
        """Why this molecule is out of scope, or None when it is in scope."""
        if self.require_single_component and len(Chem.GetMolFrags(molecule)) > 1:
            return "several disconnected components (a salt or a mixture)"
        if self.require_carbon and not any(a.GetSymbol() == "C" for a in molecule.GetAtoms()):
            return "no carbon: inorganic"
        if self.require_neutral and any(a.GetFormalCharge() != 0 for a in molecule.GetAtoms()):
            return "carries a formal charge: ionic"

        weight = Descriptors.MolWt(molecule)
        if not self.min_weight <= weight <= self.max_weight:
            return (
                f"molecular weight {weight:.0f} outside {self.min_weight:.0f}-{self.max_weight:.0f}"
            )
        return None

    def contains(self, molecules: Sequence[Mol]) -> np.ndarray:
        """Boolean mask: True where the model may be asked at all."""
        return np.array([self.reason_for_rejection(m) is None for m in molecules], dtype=bool)


def tanimoto_similarity(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Pairwise Tanimoto between two sets of binary fingerprints.

    ``|A and B| / |A or B|``, computed as a matrix product: the product counts
    shared bits, and the union follows from the two population counts.

    Returns:
        Matrix of shape ``(len(query), len(reference))``. Molecules with no bits at
        all score 0 against everything, including themselves -- there is no evidence
        of similarity to be had.
    """
    left = (np.asarray(query) > 0).astype(np.float32)
    right = (np.asarray(reference) > 0).astype(np.float32)
    if left.shape[1] != right.shape[1]:
        raise ValueError(f"Fingerprint widths differ: {left.shape[1]} against {right.shape[1]}")

    shared = left @ right.T
    union = left.sum(axis=1)[:, None] + right.sum(axis=1)[None, :] - shared
    return np.divide(shared, union, out=np.zeros_like(shared), where=union > 0)


class ApplicabilityDomain:
    """Flags molecules too unlike the training set for the model to be trusted on."""

    def __init__(self, threshold: float | None = None) -> None:
        self.threshold = threshold
        self.reference_: np.ndarray | None = None

    def fit(self, fingerprints: np.ndarray) -> ApplicabilityDomain:
        """Remember the training fingerprints. Nothing is learned beyond that."""
        self.reference_ = (np.asarray(fingerprints) > 0).astype(np.float32)
        return self

    def max_similarity(self, fingerprints: np.ndarray) -> np.ndarray:
        """Similarity to the closest training molecule, one value per query."""
        if self.reference_ is None:
            raise RuntimeError("Call fit before max_similarity")
        if len(fingerprints) == 0:
            return np.zeros(0, dtype=np.float32)
        return tanimoto_similarity(fingerprints, self.reference_).max(axis=1)

    def calibrate(
        self, fingerprints: np.ndarray, *, coverage: float = DEFAULT_COVERAGE
    ) -> ApplicabilityDomain:
        """Pick the threshold that keeps *coverage* of a held-out set inside.

        The threshold is a policy, not a fact about chemistry, so it is set on the
        validation split rather than guessed. ``coverage=0.95`` means: accept the
        95% of ordinary molecules, refuse the 5% least familiar. Refusing more
        raises quality on what remains and answers fewer questions; that trade is
        the operator's to make, which is why it is a parameter.
        """
        if not 0.0 < coverage <= 1.0:
            raise ValueError(f"Coverage must be in (0, 1], got {coverage}")
        similarity = self.max_similarity(fingerprints)
        self.threshold = float(np.quantile(similarity, 1.0 - coverage))
        return self

    def is_inside(self, fingerprints: np.ndarray) -> np.ndarray:
        """True where the model may be trusted, False where it should refuse."""
        if self.threshold is None:
            raise RuntimeError("Call calibrate or pass a threshold before is_inside")
        return self.max_similarity(fingerprints) >= self.threshold
