"""A feature matrix that carries its column names."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class FeatureMatrix:
    """Features for a set of molecules, with the name of every column.

    Names travel with the values on purpose. Without them, the question "which
    feature made the model say vanilla?" -- the whole point of phase 3 -- can only
    be answered by re-deriving the column order by hand, which is how off-by-one
    interpretation bugs happen.
    """

    values: np.ndarray
    names: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError(f"Expected a 2-D matrix, got shape {self.values.shape}")
        if self.values.shape[1] != len(self.names):
            raise ValueError(
                f"{self.values.shape[1]} columns but {len(self.names)} names: "
                "the matrix and its labels disagree"
            )

    @property
    def shape(self) -> tuple[int, int]:
        rows, columns = self.values.shape
        return rows, columns

    def hstack(self, other: FeatureMatrix) -> FeatureMatrix:
        """Concatenate side by side, keeping names aligned with columns."""
        if self.values.shape[0] != other.values.shape[0]:
            raise ValueError(
                f"Cannot stack {self.values.shape[0]} rows with {other.values.shape[0]}"
            )
        return FeatureMatrix(
            values=np.hstack([self.values, other.values]),
            names=self.names + other.names,
        )
