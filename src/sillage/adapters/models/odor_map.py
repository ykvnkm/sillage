"""The odour map built by ``scripts/build_map.py``, and placing new molecules on it.

UMAP ran offline; what ships is its result -- a 2-D position for every reference
molecule -- plus the embeddings it was computed from. A new molecule is placed at the
similarity-weighted mean position of its nearest references in embedding space. The
build script measures how far that lands from UMAP's own placement; see journal E-016.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from sillage.domain.families import Family
from sillage.domain.odor_map import MapPoint, MapPosition

MAP_FILE = "map.npz"
NEIGHBOURS = 10


class EmbeddingMap:
    """``OdorMap`` over precomputed positions and embeddings."""

    def __init__(
        self,
        *,
        smiles: np.ndarray,
        coords: np.ndarray,
        embeddings: np.ndarray,
        families: np.ndarray,
        family_order: np.ndarray,
    ) -> None:
        if not len(smiles) == len(coords) == len(embeddings) == len(families):
            raise ValueError("Map arrays disagree in length: the file is corrupt")
        self._smiles = [str(s) for s in smiles]
        self._coords = np.asarray(coords, dtype=np.float64)
        unit = np.asarray(embeddings, dtype=np.float64)
        self._unit = unit / np.maximum(np.linalg.norm(unit, axis=1, keepdims=True), 1e-12)
        order = [Family(str(f)) for f in family_order]
        self._families = [order[int(i)] for i in families]

    @classmethod
    def from_file(cls, path: Path) -> EmbeddingMap:
        with np.load(path) as data:
            return cls(**{key: data[key] for key in data.files})

    def __len__(self) -> int:
        return len(self._smiles)

    @property
    def dimensions(self) -> int:
        return int(self._unit.shape[1])

    def locate(self, embedding: Sequence[float]) -> MapPosition:
        x, y = self.locate_vector(np.asarray(embedding, dtype=np.float64))
        return MapPosition(x=float(x), y=float(y))

    def locate_vector(self, embedding: np.ndarray) -> np.ndarray:
        if embedding.shape != (self.dimensions,):
            raise ValueError(
                f"The map was built from {self.dimensions}-dimensional embeddings, "
                f"got {embedding.shape}: map and model are out of step"
            )
        query = embedding / max(float(np.linalg.norm(embedding)), 1e-12)
        similarity = self._unit @ query
        k = min(NEIGHBOURS, similarity.size)
        nearest = (
            np.arange(similarity.size)
            if k == similarity.size
            else np.argpartition(-similarity, k)[:k]
        )
        # Squared cosine similarity: the closest few dominate, a far tail barely moves it.
        weights = np.clip(similarity[nearest], 0.0, None) ** 2
        if weights.sum() <= 0.0:
            weights = np.ones_like(weights)
        return (self._coords[nearest] * weights[:, None]).sum(axis=0) / weights.sum()

    def points(self) -> tuple[MapPoint, ...]:
        return tuple(
            MapPoint(smiles=s, position=MapPosition(x=float(x), y=float(y)), family=f)
            for s, (x, y), f in zip(self._smiles, self._coords, self._families, strict=True)
        )
