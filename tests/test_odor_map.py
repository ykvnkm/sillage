"""The odour map: families cover every descriptor, new molecules land where they should."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from sillage.adapters.models.odor_map import EmbeddingMap
from sillage.data.schemas import OPENPOM_DESCRIPTORS
from sillage.domain.families import FAMILY_NAMES_RU, FAMILY_OF, Family, dominant_family
from sillage.interfaces.api.app import create_app
from sillage.interfaces.composition import build_services


def test_every_descriptor_has_exactly_one_family() -> None:
    assert set(FAMILY_OF) == set(OPENPOM_DESCRIPTORS)


def test_eight_named_families_and_other() -> None:
    """More than eight categories cannot share one chart and stay distinguishable."""
    assert len([f for f in Family if f is not Family.OTHER]) == 8
    assert set(FAMILY_NAMES_RU) == set(Family)


def test_a_rare_specific_note_outweighs_a_common_vague_one() -> None:
    frequency = {"sweet": 0.30, "vanilla": 0.02, "floral": 0.23}

    assert dominant_family(["sweet", "vanilla"], frequency) is Family.GOURMAND
    assert dominant_family(["floral", "sweet"], frequency) is Family.FLORAL


def test_only_unclassifiable_notes_make_other() -> None:
    assert dominant_family(["odorless"], {"odorless": 0.05}) is Family.OTHER
    assert dominant_family([], {}) is Family.OTHER


# --- placement ------------------------------------------------------------------------


def grid_map() -> EmbeddingMap:
    embeddings = np.eye(4)
    return EmbeddingMap(
        smiles=np.array(["a", "b", "c", "d"]),
        coords=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        embeddings=embeddings,
        families=np.array([0, 1, 2, 3], dtype=np.int8),
        family_order=np.array([f.value for f in Family]),
    )


def test_a_reference_embedding_lands_on_its_own_point() -> None:
    position = grid_map().locate((0.0, 0.0, 1.0, 0.0))

    assert (position.x, position.y) == pytest.approx((0.0, 1.0))


def test_a_mixed_embedding_lands_between_its_references() -> None:
    position = grid_map().locate((1.0, 1.0, 0.0, 0.0))

    assert (position.x, position.y) == pytest.approx((0.5, 0.0))


def test_a_map_from_another_model_is_refused() -> None:
    with pytest.raises(ValueError, match="out of step"):
        grid_map().locate((1.0, 0.0))


def test_points_carry_their_families() -> None:
    families = [p.family for p in grid_map().points()]

    assert families == [Family.FRUITY, Family.CITRUS, Family.FLORAL, Family.GREEN]


# --- API ------------------------------------------------------------------------------


@pytest.fixture
def client(artifact: Path):
    with TestClient(create_app(lambda: build_services(artifact))) as test_client:
        yield test_client


def test_map_endpoint_returns_every_reference_molecule(client: TestClient) -> None:
    body = client.get("/map").json()

    points = body["points"]
    n = len(points["smiles"])
    assert n > 0
    assert len(points["x"]) == len(points["y"]) == len(points["family"]) == len(points["name"]) == n
    assert all(0.0 <= v <= 1.0 for v in points["x"] + points["y"])
    assert [f["id"] for f in body["families"]][-1] == "other"
    assert points["name"][points["smiles"].index("CCCCO")] == "бутанол"


def test_map_is_compressed_on_the_wire(client: TestClient) -> None:
    response = client.get("/map", headers={"Accept-Encoding": "gzip"})

    assert response.headers.get("content-encoding") == "gzip"


def test_prediction_says_where_the_molecule_sits(client: TestClient) -> None:
    position = client.post("/predict", json={"query": "CCCCCCO"}).json()["map_position"]

    assert 0.0 <= position["x"] <= 1.0
    assert 0.0 <= position["y"] <= 1.0
