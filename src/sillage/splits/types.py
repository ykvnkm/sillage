"""The result of splitting a dataset."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PARTS = ("train", "val", "test")


@dataclass(frozen=True, slots=True)
class Split:
    """Row indices for the three parts of a dataset.

    Indices rather than copies of the data: the same split has to be reused by
    every model in phases 1 and 2, and comparing models on "the same split" only
    means anything if it is literally the same object, not a re-derived one.
    """

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    strategy: str
    seed: int

    def __post_init__(self) -> None:
        parts = [self.train, self.val, self.test]
        if any(part.ndim != 1 for part in parts):
            raise ValueError("Each part must be a 1-D array of row indices")

        combined = np.concatenate(parts)
        unique = np.unique(combined)
        if unique.size != combined.size:
            raise ValueError("The parts overlap: some row appears in more than one part")
        if unique.size and not np.array_equal(unique, np.arange(unique.size)):
            raise ValueError("The parts do not cover 0..n-1 exactly once")

    @property
    def sizes(self) -> dict[str, int]:
        return {"train": self.train.size, "val": self.val.size, "test": self.test.size}

    @property
    def n_rows(self) -> int:
        return self.train.size + self.val.size + self.test.size

    def parts(self) -> dict[str, np.ndarray]:
        return {"train": self.train, "val": self.val, "test": self.test}


def label_coverage(labels: np.ndarray, split: Split) -> np.ndarray:
    """Positive examples of every label in every part.

    Returns:
        Array of shape ``(n_labels, 3)`` in train/val/test order. A zero anywhere
        means a label whose metric cannot be computed on that part.
    """
    return np.column_stack([labels[indices].sum(axis=0) for indices in split.parts().values()])
