"""The odour map: where molecules sit in the model's space of smells, flattened to 2-D."""

from __future__ import annotations

from dataclasses import dataclass

from sillage.domain.families import Family


@dataclass(frozen=True, slots=True)
class MapPosition:
    """A point on the map; both coordinates in 0..1."""

    x: float
    y: float


@dataclass(frozen=True, slots=True)
class MapPoint:
    """A reference molecule on the map, with the family experts' labels put it in."""

    smiles: str
    position: MapPosition
    family: Family
