"""A model of several components (journal E-020): saved, loaded, averaged, served.

The graph networks here have random weights -- what is under test is the wiring, and
random weights need no torch.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from conftest import LABELS, SMILES, _labels
from sillage.adapters.chem import RdkitMoleculeReader
from sillage.adapters.models import ArtifactOdorPredictor
from sillage.features import parse_smiles
from sillage.features import sets as feature_sets
from sillage.features.graphs import ATOM_FEATURES, BOND_FEATURES, molecular_graph
from sillage.models.artifact import ArtifactError, OdorModel
from sillage.models.graph_network import GraphEnsemble, GraphNetwork
from sillage.research import train_v2

HIDDEN = 8
R_CARVONE = "CC1=CC[C@@H](CC1=O)C(C)=C"
S_CARVONE = "CC1=CC[C@H](CC1=O)C(C)=C"


def random_network(seed: int) -> GraphNetwork:
    rng = np.random.default_rng(seed)
    shapes = {
        "bond_in.weight": (HIDDEN, ATOM_FEATURES + BOND_FEATURES),
        "bond_step.weight": (HIDDEN, HIDDEN),
        "atom_out.weight": (HIDDEN, ATOM_FEATURES + HIDDEN),
        "atom_out.bias": (HIDDEN,),
        "hidden_layer.weight": (HIDDEN, HIDDEN),
        "hidden_layer.bias": (HIDDEN,),
        "output.weight": (len(LABELS), HIDDEN),
        "output.bias": (len(LABELS),),
    }
    return GraphNetwork({k: rng.normal(0, 0.5, s).astype(np.float32) for k, s in shapes.items()}, 3)


@pytest.fixture(scope="module")
def ensemble(artifact) -> OdorModel:
    linear = OdorModel.load(artifact)
    descriptors = feature_sets.build("descriptors", parse_smiles(SMILES))
    y = np.array([_labels(s) for s in SMILES], dtype=float)
    boosting = train_v2.fit_boosting(descriptors, y, np.arange(len(SMILES)))
    card = replace(
        linear.card,
        name="mini-ensemble",
        components=("linear", "boosting", "graph"),
        boosting_feature_set="descriptors",
    )
    graph = GraphEnsemble((random_network(0), random_network(1)))
    return replace(linear, card=card, boosting=boosting, graph=graph)


def test_raw_score_is_the_mean_of_the_components(ensemble: OdorModel) -> None:
    molecules = parse_smiles(["CCO", "Oc1ccccc1"])
    features = feature_sets.build("descriptors", molecules).values

    expected = np.mean(
        [
            ensemble.estimator.predict_proba(features),
            ensemble.boosting.predict_proba(features),
            np.stack([ensemble.graph.predict(molecular_graph(m)) for m in molecules]),
        ],
        axis=0,
    )

    np.testing.assert_allclose(ensemble.scorer().raw(molecules), expected, atol=1e-9)


def test_round_trip_keeps_every_component(ensemble: OdorModel, tmp_path) -> None:
    molecules = parse_smiles(["CCO", "Oc1ccccc1", R_CARVONE])

    loaded = OdorModel.load(ensemble.save(tmp_path / "m"))

    assert loaded.card.components == ("linear", "boosting", "graph")
    np.testing.assert_allclose(
        loaded.scorer().raw(molecules), ensemble.scorer().raw(molecules), atol=1e-6
    )


def test_a_listed_component_that_is_missing_refuses_to_load(ensemble: OdorModel, tmp_path) -> None:
    directory = ensemble.save(tmp_path / "m")
    (directory / "graph.npz").unlink()

    with pytest.raises(ArtifactError, match="graph"):
        OdorModel.load(directory)


def test_the_served_model_tells_enantiomers_apart(ensemble: OdorModel) -> None:
    predictor = ArtifactOdorPredictor(ensemble)
    reader = RdkitMoleculeReader()

    r = predictor.embed(reader.read(R_CARVONE))
    s = predictor.embed(reader.read(S_CARVONE))

    assert predictor.sees_stereo
    assert r != s


def test_a_model_without_graph_networks_cannot_see_stereo(artifact) -> None:
    predictor = ArtifactOdorPredictor(OdorModel.load(artifact))
    reader = RdkitMoleculeReader()

    assert not predictor.sees_stereo
    assert predictor.embed(reader.read(R_CARVONE)) == predictor.embed(reader.read(S_CARVONE))
