"""Atom highlighting (journal E-021): occlusion that means the same for every component."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pytest
from rdkit import Chem

from conftest import LABELS, SMILES, _labels
from sillage.adapters.chem import RdkitMoleculeReader
from sillage.adapters.chem.depict import RdkitDepicter
from sillage.adapters.models import ArtifactOdorPredictor
from sillage.application import ExplainNote
from sillage.domain import InvalidMolecule, NoteExplanation, Refusal, RefusalReason
from sillage.features import parse_smiles
from sillage.features import sets as feature_sets
from sillage.features.graphs import molecular_graph
from sillage.models import BlockScaledLogisticRegression, MultilabelClassifier
from sillage.models.artifact import OdorModel
from sillage.models.attribution import (
    NotAttributable,
    _bits_owned_by_each_atom,
    explain,
    without_atom,
)

FEATURES = "morgan-bin-r3+desc"


@pytest.fixture(scope="module")
def fingerprint_model(artifact) -> OdorModel:
    """The mini artifact, refitted on the fingerprint feature set atoms can be traced in."""
    base = OdorModel.load(artifact)
    matrix = feature_sets.build(FEATURES, parse_smiles(SMILES))
    y = np.array([_labels(s) for s in SMILES], dtype=np.int8)
    estimator = MultilabelClassifier(BlockScaledLogisticRegression(feature_names=matrix.names))
    estimator.fit(matrix.values, y)
    return replace(base, card=replace(base.card, feature_set=FEATURES), estimator=estimator)


def test_removing_an_atom_removes_its_bonds_and_renumbers_the_rest() -> None:
    graph = molecular_graph(Chem.MolFromSmiles("CCO"))  # 0-1, 1-2

    without_middle = without_atom(graph, 1)
    without_end = without_atom(graph, 0)

    assert without_middle.n_atoms == 2 and without_middle.bonds.shape == (0, 2)
    assert without_end.bonds.tolist() == [[0, 1]]  # old 1-2, renumbered
    np.testing.assert_array_equal(without_end.atoms, graph.atoms[1:])


def test_a_bit_set_elsewhere_too_survives_removing_one_atom() -> None:
    """Both methyls of propane set the same radius-0 bit: neither owns it alone."""
    molecule = Chem.MolFromSmiles("CCC")
    owned = _bits_owned_by_each_atom(molecule, radius=3, n_bits=2048)
    from rdkit.Chem import rdFingerprintGenerator

    output = rdFingerprintGenerator.AdditionalOutput()
    output.AllocateBitInfoMap()
    rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048).GetFingerprintAsNumPy(
        molecule, additionalOutput=output
    )
    methyl_bit = next(
        bit for bit, envs in output.GetBitInfoMap().items() if envs == ((0, 0), (2, 0))
    )

    assert methyl_bit not in owned[0] and methyl_bit not in owned[2]
    assert owned[1]  # the middle carbon owns the bits only it centres


def test_sulfur_carries_the_sulfurous_note(fingerprint_model: OdorModel) -> None:
    molecule = Chem.MolFromSmiles("CCSCC")
    label = LABELS.index("sulfurous")

    (found,) = explain(fingerprint_model.scorer(), fingerprint_model.calibrator, molecule, [label])

    assert found.atoms.shape == (5,)
    assert int(np.argmax(found.atoms)) == 2


def test_contributions_start_from_the_served_probability(fingerprint_model: OdorModel) -> None:
    predictor = ArtifactOdorPredictor(fingerprint_model)
    molecule = RdkitMoleculeReader().read("Oc1ccccc1")

    explanation = predictor.explain(molecule, "aromatic")
    served = {d.name: d for d in predictor.predict(molecule).scores}["aromatic"]

    assert predictor.explains
    assert explanation.smiles == molecule.model_smiles
    assert len(explanation.atoms) == 7
    assert explanation.probability == pytest.approx(served.score, abs=1e-9)


def test_unknown_descriptor_is_a_key_error(fingerprint_model: OdorModel) -> None:
    with pytest.raises(KeyError):
        ArtifactOdorPredictor(fingerprint_model).explain(RdkitMoleculeReader().read("CCO"), "x")


def test_models_without_traceable_features_do_not_explain(artifact) -> None:
    model = OdorModel.load(artifact)  # descriptors only: no bit belongs to an atom

    assert not ArtifactOdorPredictor(model).explains
    with pytest.raises(NotAttributable):
        explain(model.scorer(), model.calibrator, Chem.MolFromSmiles("CCO"), [0])


# --- use case ---------------------------------------------------------------------


@dataclass
class StubExplainer:
    def explain(self, molecule, descriptor: str) -> NoteExplanation:
        if descriptor != "fruity":
            raise KeyError(descriptor)
        return NoteExplanation(descriptor, 0.9, molecule.model_smiles, (0.1,) * 3)


class StubScope:
    def check(self, molecule):
        if "Br" in molecule.smiles:
            return Refusal(RefusalReason.UNSEEN_ELEMENT, "contains Br", molecule)
        return None


@pytest.fixture
def explain_note() -> ExplainNote:
    return ExplainNote(reader=RdkitMoleculeReader(), scope=StubScope(), explainer=StubExplainer())


def test_explaining_answers_with_the_molecule_as_the_model_read_it(explain_note) -> None:
    result = explain_note(" CCO ", "fruity")

    assert isinstance(result, NoteExplanation)
    assert result.smiles == "CCO"


@pytest.mark.parametrize(
    ("smiles", "descriptor", "reason"),
    [
        ("C1CC", "fruity", RefusalReason.INVALID_INPUT),
        ("CCBr", "fruity", RefusalReason.UNSEEN_ELEMENT),
        ("CCO", "nonsense", RefusalReason.INVALID_INPUT),
        ("", "fruity", RefusalReason.INVALID_INPUT),
    ],
)
def test_explaining_refuses_what_predicting_refuses(explain_note, smiles, descriptor, reason):
    result = explain_note(smiles, descriptor)

    assert isinstance(result, Refusal)
    assert result.reason is reason


# --- drawing ----------------------------------------------------------------------


def test_shaded_drawing_differs_from_the_plain_one() -> None:
    depicter = RdkitDepicter()

    plain = depicter.svg("CCSCC", 300, 200)
    shaded = depicter.svg("CCSCC", 300, 200, (0.0, 0.02, 0.3, 0.02, -0.1))

    assert plain != shaded
    assert "#D08931" in shaded  # the strongest atom: full amber, as RDKit rounds it
    assert "#C2D1DF" in shaded  # the one holding the note back: a third-strength blue


def test_weights_must_match_the_atoms() -> None:
    with pytest.raises(InvalidMolecule):
        RdkitDepicter().svg("CCO", 300, 200, (0.1, 0.2))


# --- HTTP -------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fingerprint_client(fingerprint_model: OdorModel, tmp_path_factory):
    from fastapi.testclient import TestClient

    from sillage.interfaces.api.app import create_app
    from sillage.interfaces.composition import build_services

    directory = fingerprint_model.save(tmp_path_factory.mktemp("fp") / "fp")
    with TestClient(create_app(lambda: build_services(directory))) as client:
        yield client


def test_explain_returns_one_number_per_atom_and_a_drawing(fingerprint_client) -> None:
    response = fingerprint_client.post(
        "/explain", json={"smiles": "CCSCC", "descriptor": "sulfurous"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["smiles"] == "CCSCC"
    assert len(body["atoms"]) == 5
    assert body["probability"] is None  # sulfurous is uncalibrated in the mini model
    assert body["svg"].startswith("<svg")


def test_explain_refuses_in_the_shape_predict_does(fingerprint_client) -> None:
    response = fingerprint_client.post("/explain", json={"smiles": "CCBr", "descriptor": "x"})

    assert response.status_code == 422
    assert response.json()["reason"] == "unseen_element"


def test_a_model_that_cannot_explain_says_so(artifact) -> None:
    from fastapi.testclient import TestClient

    from sillage.interfaces.api.app import create_app
    from sillage.interfaces.composition import build_services

    with TestClient(create_app(lambda: build_services(artifact))) as client:
        response = client.post("/explain", json={"smiles": "CCO", "descriptor": "oxygenated"})

    assert response.status_code == 404
