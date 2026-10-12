"""The HTTP contract, end to end on the miniature artifact.

What a caller relies on: status codes, one error shape for every kind of refusal,
no number for an uncalibrated descriptor, and metrics that count what they claim.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from sillage.interfaces.api.app import create_app
from sillage.interfaces.composition import build_services


@pytest.fixture
def client(artifact: Path) -> Iterator[TestClient]:
    with TestClient(create_app(lambda: build_services(artifact))) as test_client:
        yield test_client


def test_predict_answers_with_the_documented_shape(client: TestClient) -> None:
    response = client.post("/predict", json={"smiles": "CCCCO", "neighbours": 2})

    assert response.status_code == 200
    body = response.json()
    assert body["refused"] is False
    assert body["model"] == "mini"
    assert body["molecule"]["smiles"] == "CCCCO"
    assert body["known"] == {"descriptors": ["oxygenated"], "seen_in_training": True}
    assert len(body["neighbours"]) == 2
    assert {"descriptor", "confidence", "probability", "base_rate"} <= set(body["predicted"][0])


def test_uncalibrated_descriptor_never_carries_a_number(client: TestClient) -> None:
    body = client.post("/predict", json={"smiles": "CCSCC"}).json()

    assert [d["descriptor"] for d in body["unverified"]] == ["sulfurous"]
    assert all(d["probability"] is None for d in body["unverified"])
    assert all(d["probability"] is not None for d in body["predicted"])
    assert "score" not in str(body)


def test_stereo_input_is_answered_with_a_caveat(client: TestClient) -> None:
    body = client.post("/predict", json={"smiles": "C[C@H](O)c1ccccc1"}).json()

    assert body["caveats"] == ["stereo_discarded"]
    assert body["molecule"]["input_smiles"] == "C[C@H](O)c1ccccc1"


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ({"smiles": "BrCCCC"}, "unseen_element"),
        ({"smiles": "[Na+].[Cl-]"}, "multiple_components"),
        ({"smiles": "hello"}, "unknown_name"),
        ({"query": "C1CC"}, "invalid_input"),
        # Schema-level failures land in the same shape as domain refusals.
        ({"smiles": ""}, "invalid_input"),
        ({"smiles": "C" * 501}, "invalid_input"),
        ({"smiles": "CCO", "neighbours": 1000}, "invalid_input"),
        ({}, "invalid_input"),
    ],
)
def test_every_refusal_is_422_with_one_body_shape(
    client: TestClient, payload: dict, reason: str
) -> None:
    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    body = response.json()
    assert body["refused"] is True
    assert body["reason"] == reason
    assert body["detail"]


def test_model_endpoint_reports_metrics_next_to_their_floor(client: TestClient) -> None:
    body = client.get("/model").json()

    assert body["name"] == "mini"
    assert body["descriptors"] == ["oxygenated", "aromatic", "sulfurous"]
    assert body["unverified"] == ["sulfurous"]
    assert "C, N, O, S" in body["scope"]
    assert set(body) >= {"metrics", "floor", "trained_on"}


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok", "model": "mini"}


def test_metrics_count_answers_refusals_and_reasons(client: TestClient) -> None:
    client.post("/predict", json={"smiles": "CCCCO"})
    client.post("/predict", json={"smiles": "BrCCCC"})
    client.post("/predict", json={"smiles": ""})

    text = client.get("/metrics").text

    assert 'sillage_predictions_total{outcome="answered"} 1.0' in text
    assert 'sillage_predictions_total{outcome="refused"} 2.0' in text
    assert 'sillage_refusals_total{reason="unseen_element"} 1.0' in text
    assert 'sillage_refusals_total{reason="invalid_input"} 1.0' in text
    assert 'sillage_known_molecules_total{seen_in_training="true"} 1.0' in text
    assert "sillage_nearest_similarity_count 1.0" in text
    assert 'sillage_model_info{model="mini"} 1.0' in text


def test_metrics_label_routes_by_template_not_by_raw_path(client: TestClient) -> None:
    """A label per distinct URL is unbounded cardinality; scanners would create
    thousands of series by requesting random paths."""
    client.get("/no-such-page-1")
    client.get("/no-such-page-2")

    text = client.get("/metrics").text

    assert "no-such-page" not in text
    assert 'route="unmatched",status="404"} 2.0' in text


def test_each_app_has_its_own_counters(artifact: Path) -> None:
    load = lambda: build_services(artifact)  # noqa: E731
    with TestClient(create_app(load)) as first:
        first.post("/predict", json={"smiles": "CCCCO"})
    with TestClient(create_app(load)) as second:
        text = second.get("/metrics").text

    assert 'sillage_predictions_total{outcome="answered"}' not in text


# --- web interface and drawings ---------------------------------------------------


def test_root_serves_the_web_interface(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert 'id="query-form"' in response.text


@pytest.mark.parametrize("path", ["/static/app.css", "/static/app.js", "/static/favicon.svg"])
def test_static_assets_are_served(client: TestClient, path: str) -> None:
    assert client.get(path).status_code == 200


def test_depict_returns_a_cacheable_svg(client: TestClient) -> None:
    response = client.get("/depict", params={"smiles": "COc1cc(C=O)ccc1O", "width": 300})

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/svg+xml"
    assert "max-age" in response.headers["cache-control"]
    assert response.text.lstrip().startswith("<svg")
    assert "viewBox='0 0 300 240'" in response.text


def test_depict_draws_in_one_ink(client: TestClient) -> None:
    """Colour means confidence on this page; a red oxygen would compete with it."""
    import re

    svg = client.get("/depict", params={"smiles": "OCC(=O)N"}).text

    assert set(re.findall(r"#[0-9A-Fa-f]{6}", svg)) == {"#25180F"}


@pytest.mark.parametrize(
    "params",
    [{"smiles": "hello"}, {"smiles": ""}, {"smiles": "CCO", "width": 5000}, {}],
)
def test_depict_refuses_like_predict(client: TestClient, params: dict) -> None:
    response = client.get("/depict", params=params)

    assert response.status_code == 422
    assert response.json()["refused"] is True
    assert response.json()["reason"] == "invalid_input"


# --- names -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "kind"),
    [
        ("бутанол", "name"),
        ("1-Butanol", "name"),
        ("n-butyl alcohol", "name"),
        ("CCCCO", "smiles"),
        ("OCCCC", "smiles"),
    ],
)
def test_a_molecule_is_found_by_any_of_its_names_or_its_structure(
    client: TestClient, query: str, kind: str
) -> None:
    body = client.post("/predict", json={"query": query}).json()

    assert body["molecule"]["smiles"] == "CCCCO"
    assert body["names"]["display"] == "бутанол"
    assert body["names"]["en"] == "1-Butanol"
    assert body["names"]["pubchem_url"] == "https://pubchem.ncbi.nlm.nih.gov/compound/263"
    assert body["query_kind"] == kind


def test_a_transliterated_name_finds_the_english_one(client: TestClient) -> None:
    """No Russian name is stored for diethyl sulfide; the folded skeleton still matches."""
    body = client.post("/predict", json={"query": "диэтилсульфид"}).json()

    assert body["molecule"]["smiles"] == "CCSCC"
    assert body["names"]["display"] == "Diethyl sulfide"
    assert body["matched_name"] == "Diethyl sulfide"


def test_an_alias_is_reported_as_the_matched_name(client: TestClient) -> None:
    """A trade name the user typed leads the page, not the systematic title."""
    body = client.post("/predict", json={"query": "ethyl sulfide"}).json()

    assert body["matched_name"] == "Ethyl sulfide"
    assert body["names"]["en"] == "Diethyl sulfide"


def test_neighbours_carry_their_names(client: TestClient) -> None:
    body = client.post("/predict", json={"query": "бензол", "neighbours": 20}).json()

    named = {n["smiles"]: n["names"] for n in body["neighbours"] if n["names"]}
    assert named["O=Cc1ccccc1"]["display"] == "Benzaldehyde"


def test_suggest_matches_russian_and_english_prefixes(client: TestClient) -> None:
    russian = client.get("/suggest", params={"q": "бенз"}).json()
    english = client.get("/suggest", params={"q": "benz"}).json()

    assert {s["smiles"] for s in russian} == {"c1ccccc1", "O=Cc1ccccc1"}
    assert russian == english
    # The molecule whose primary name matches comes before one matched by an alias.
    assert russian[0]["name"] == "бензол"


def test_suggest_with_nothing_matching_is_empty(client: TestClient) -> None:
    assert client.get("/suggest", params={"q": "zzzz"}).json() == []


def test_unknown_name_offline_is_a_clear_refusal(client: TestClient) -> None:
    response = client.post("/predict", json={"query": "galaxolide"})

    assert response.status_code == 422
    assert response.json()["reason"] == "unknown_name"


def test_metrics_publish_the_similarity_baseline_for_drift(client: TestClient) -> None:
    text = client.get("/metrics").text

    for quantile in ("0.1", "0.25", "0.5"):
        assert f'sillage_baseline_nearest_similarity{{quantile="{quantile}"}}' in text
