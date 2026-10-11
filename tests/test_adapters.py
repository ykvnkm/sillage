"""Adapters and the composition root, on a miniature but real artifact.

Real RDKit, real feature set, real classifier and calibrator -- only small. The stubs
in ``test_predict_odor.py`` prove the rules; these prove the wiring delivers them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sillage.adapters.chem import FingerprintReferenceLibrary, RdkitMoleculeReader
from sillage.adapters.models import ArtifactOdorPredictor, ArtifactScopePolicy
from sillage.domain import (
    Caveat,
    InvalidMolecule,
    KnownMolecule,
    OdorReport,
    Refusal,
    RefusalReason,
)
from sillage.features import parse_smiles
from sillage.features import sets as feature_sets
from sillage.interfaces import cli
from sillage.interfaces.composition import build_predict_odor
from sillage.models import (
    ChemicalScope,
)
from sillage.models.artifact import ArtifactError, OdorModel, ReferenceMolecule

# --- reader ----------------------------------------------------------------------


def test_reader_canonicalises_and_drops_stereo() -> None:
    left = RdkitMoleculeReader().read("C[C@H](O)CC")
    right = RdkitMoleculeReader().read("CC[C@@H](C)O")

    assert left.smiles == right.smiles == "CCC(C)O"
    assert left.stereo_discarded and right.stereo_discarded
    assert left.input_smiles == "C[C@H](O)CC"
    assert left.formula == "C4H10O"
    assert left.weight == pytest.approx(74.12, abs=0.01)


def test_reader_does_not_invent_stereo_caveats() -> None:
    assert not RdkitMoleculeReader().read("OCC").stereo_discarded


@pytest.mark.parametrize("text", ["hello", "C1CC", "C(C"])
def test_reader_rejects_what_is_not_a_molecule(text: str) -> None:
    with pytest.raises(InvalidMolecule):
        RdkitMoleculeReader().read(text)


# --- scope -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("smiles", "reason"),
    [
        ("[Na+].[Cl-]", RefusalReason.MULTIPLE_COMPONENTS),
        ("O=O", RefusalReason.INORGANIC),
        ("C[N+](C)(C)C", RefusalReason.CHARGED),
        ("BrCCCC", RefusalReason.UNSEEN_ELEMENT),
        ("C" * 40, RefusalReason.WEIGHT_OUT_OF_RANGE),
    ],
)
def test_scope_policy_returns_a_stable_code(smiles: str, reason: RefusalReason) -> None:
    policy = ArtifactScopePolicy(ChemicalScope(allowed_elements=("C", "N", "O", "S")))

    refusal = policy.check(RdkitMoleculeReader().read(smiles))

    assert refusal is not None
    assert refusal.reason is reason
    assert refusal.detail


def test_scope_policy_accepts_an_ordinary_molecule() -> None:
    policy = ArtifactScopePolicy(ChemicalScope(allowed_elements=("C", "O")))

    assert policy.check(RdkitMoleculeReader().read("CCCC=O")) is None


# --- predictor -------------------------------------------------------------------


def test_predictor_matches_the_artifact_it_wraps(artifact: Path) -> None:
    """The adapter must add nothing and lose nothing: same numbers as the model."""
    model = OdorModel.load(artifact)
    profile = ArtifactOdorPredictor(model).predict(RdkitMoleculeReader().read("CCCCO"))

    expected = model.probabilities(
        feature_sets.build("descriptors", parse_smiles(["CCCCO"])).values
    )[0]
    assert [s.name for s in profile.scores] == list(model.label_names)
    assert [s.score for s in profile.scores] == pytest.approx(expected.tolist())


def test_predictor_marks_labels_the_calibrator_refused(artifact: Path) -> None:
    profile = ArtifactOdorPredictor(OdorModel.load(artifact)).predict(
        RdkitMoleculeReader().read("CCSCC")
    )

    assert profile["oxygenated"].calibrated
    assert not profile["sulfurous"].calibrated
    assert profile["sulfurous"].probability is None


def test_predictor_refuses_to_serve_an_uncalibrated_model(artifact: Path) -> None:
    model = OdorModel.load(artifact)
    bare = OdorModel(
        card=model.card,
        estimator=model.estimator,
        calibrator=None,
        scope=model.scope,
        reference=model.reference,
    )

    with pytest.raises(ArtifactError, match="no calibrator"):
        ArtifactOdorPredictor(bare)


# --- reference library -------------------------------------------------------------


def test_lookup_finds_a_known_molecule_whatever_way_it_was_written() -> None:
    library = FingerprintReferenceLibrary(
        [ReferenceMolecule("CCCO", ("alcoholic",), "train"), ReferenceMolecule("CCC=O", (), "val")]
    )

    found = library.lookup(RdkitMoleculeReader().read("OCCC"))

    assert found == KnownMolecule(("alcoholic",), seen_in_training=True)
    assert library.lookup(RdkitMoleculeReader().read("O=CCC")) == KnownMolecule(
        (), seen_in_training=False
    )
    assert library.lookup(RdkitMoleculeReader().read("CCCCCCO")) is None


def test_nearest_excludes_the_query_and_sorts_by_similarity() -> None:
    entries = [
        ReferenceMolecule(s, (), "train") for s in ("CCCCO", "CCCCCO", "c1ccccc1", "CCCCCCO")
    ]
    library = FingerprintReferenceLibrary(entries)

    neighbours = library.nearest(RdkitMoleculeReader().read("CCCCCO"), k=3)

    assert "CCCCCO" not in [n.smiles for n in neighbours]
    assert len(neighbours) == 3
    assert [n.similarity for n in neighbours] == sorted(
        (n.similarity for n in neighbours), reverse=True
    )
    assert neighbours[-1].smiles == "c1ccccc1"


def test_nearest_with_k_zero_is_empty() -> None:
    library = FingerprintReferenceLibrary([ReferenceMolecule("CCO", (), "train")])

    assert library.nearest(RdkitMoleculeReader().read("CCO"), k=0) == ()


# --- composition root and CLI -------------------------------------------------------


def test_composed_use_case_answers_and_refuses(artifact: Path) -> None:
    predict = build_predict_odor(artifact)

    answer = predict("C[C@H](O)c1ccccc1", neighbours=2)
    assert isinstance(answer, OdorReport)
    assert answer.model_name == "mini"
    assert answer.caveats == (Caveat.STEREO_DISCARDED,)
    assert len(answer.neighbours) == 2

    refusal = predict("BrCCCC")
    assert isinstance(refusal, Refusal)
    assert refusal.reason is RefusalReason.UNSEEN_ELEMENT


def test_composition_reads_the_model_location_from_the_environment(
    artifact: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SILLAGE_MODEL_DIR", str(artifact))

    assert build_predict_odor().predictor.name == "mini"


def test_cli_exit_codes_tell_answers_from_refusals(
    artifact: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("SILLAGE_MODEL_DIR", str(artifact))

    assert cli.main(["CCCCO"]) == 0
    assert "model      mini" in capsys.readouterr().out

    assert cli.main(["BrCCCC", "--json"]) == 2
    assert '"reason": "unseen_element"' in capsys.readouterr().out


def test_json_output_never_carries_a_number_for_an_uncalibrated_label(artifact: Path) -> None:
    payload = cli.to_dict(build_predict_odor(artifact)("CCSCC"))

    assert [entry["descriptor"] for entry in payload["unverified"]] == ["sulfurous"]
    for entry in payload["predicted"]:
        assert entry["probability"] is not None
    for entry in payload["unverified"]:
        assert entry["probability"] is None
