"""The prediction use case and the domain types it returns, on stubs.

No RDKit, no model: the point of the port layer is that the rules of the product --
what gets refused, in which order, what is shown with which number -- can be tested
without either. The adapters have their own tests in ``test_adapters.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from sillage.application import MAX_INPUT_LENGTH, PredictOdor
from sillage.domain import (
    Caveat,
    Confidence,
    DescriptorScore,
    InvalidMolecule,
    KnownMolecule,
    Molecule,
    MoleculeNames,
    NameSuggestion,
    Neighbour,
    OdorProfile,
    OdorReport,
    Refusal,
    RefusalReason,
)
from sillage.domain.ports import NameLookupUnavailable


def molecule(smiles: str = "CCO", *, stereo: bool = False) -> Molecule:
    return Molecule(
        smiles=smiles, input_smiles=smiles, formula="C2H6O", weight=46.1, stereo_discarded=stereo
    )


def score(name: str, value: float, *, calibrated: bool = True) -> DescriptorScore:
    return DescriptorScore(name=name, score=value, calibrated=calibrated, base_rate=0.1)


class StubReader:
    def read(self, text: str) -> Molecule:
        if text in {"C1(C", "garbage", "ванилин", "galaxolide"}:
            raise InvalidMolecule(f"not a valid SMILES string: {text!r}")
        return molecule(text, stereo="@" in text)


@dataclass
class StubScope:
    refuse: set[str] = field(default_factory=set)

    def check(self, m: Molecule) -> Refusal | None:
        if m.smiles in self.refuse:
            return Refusal(RefusalReason.UNSEEN_ELEMENT, "contains Br", m)
        return None


@dataclass
class StubPredictor:
    calls: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return "stub"

    def predict(self, m: Molecule) -> OdorProfile:
        self.calls.append(m.smiles)
        return OdorProfile.from_scores([score("fruity", 0.9)])


class StubReferences:
    def lookup(self, m: Molecule) -> KnownMolecule | None:
        return KnownMolecule(("fruity",), seen_in_training=True) if m.smiles == "CCO" else None

    def nearest(self, m: Molecule, k: int) -> tuple[Neighbour, ...]:
        return tuple(Neighbour(f"C{'C' * i}O", 0.9 - i / 10, ("sweet",)) for i in range(k))


@pytest.fixture
def predictor() -> StubPredictor:
    return StubPredictor()


@pytest.fixture
def predict(predictor: StubPredictor) -> PredictOdor:
    return PredictOdor(
        reader=StubReader(),
        scope=StubScope(refuse={"CCBr"}),
        predictor=predictor,
        references=StubReferences(),
    )


# --- the use case ------------------------------------------------------------


def test_an_answer_carries_profile_neighbours_and_dataset_record(predict: PredictOdor) -> None:
    report = predict("CCO", neighbours=3)

    assert isinstance(report, OdorReport)
    assert report.profile["fruity"].probability == 0.9
    assert len(report.neighbours) == 3
    assert report.known == KnownMolecule(("fruity",), seen_in_training=True)
    assert report.model_name == "stub"
    assert report.caveats == ()


@pytest.mark.parametrize("text", ["", "   ", "x" * (MAX_INPUT_LENGTH + 1), "C1(C"])
def test_unreadable_input_is_refused_as_invalid(predict: PredictOdor, text: str) -> None:
    result = predict(text)

    assert isinstance(result, Refusal)
    assert result.reason is RefusalReason.INVALID_INPUT


def test_out_of_scope_molecule_never_reaches_the_model(
    predict: PredictOdor, predictor: StubPredictor
) -> None:
    """Not "predicted and then hidden": no number about it may exist at all, or
    some later code path will eventually leak it."""
    result = predict("CCBr")

    assert isinstance(result, Refusal)
    assert result.reason is RefusalReason.UNSEEN_ELEMENT
    assert result.molecule is not None
    assert predictor.calls == []


def test_discarded_stereo_is_reported_as_a_caveat(predict: PredictOdor) -> None:
    report = predict("C[C@H](N)O")

    assert isinstance(report, OdorReport)
    assert report.caveats == (Caveat.STEREO_DISCARDED,)


def test_surrounding_whitespace_is_not_part_of_the_molecule(predict: PredictOdor) -> None:
    report = predict("  CCO\n")

    assert isinstance(report, OdorReport)
    assert report.molecule.smiles == "CCO"


# --- the profile ---------------------------------------------------------------


def test_uncalibrated_score_has_no_probability_whatever_its_value() -> None:
    raw = score("cedar", 0.93, calibrated=False)

    assert raw.probability is None
    assert raw.confidence is Confidence.UNVERIFIED


def test_uncalibrated_labels_are_never_ranked_among_calibrated_ones() -> None:
    """The ambroxide case from step 2.3: sorted together, an uncalibrated `cedar` 0.59
    sat above a calibrated `musk` 0.58 while being right about half as often."""
    profile = OdorProfile.from_scores(
        [
            score("woody", 0.93),
            score("cedar", 0.59, calibrated=False),
            score("musk", 0.58),
            score("ozone", 0.30, calibrated=False),
        ]
    )

    assert [s.name for s in profile.confident()] == ["woody", "musk"]
    assert [s.name for s in profile.unverified()] == ["cedar"]


def test_confident_respects_the_requested_band() -> None:
    profile = OdorProfile.from_scores(
        [score("a", 0.90), score("b", 0.60), score("c", 0.40), score("d", 0.20)]
    )

    assert [s.name for s in profile.confident()] == ["a", "b", "c"]
    assert [s.name for s in profile.confident(Confidence.PROBABLE)] == ["a", "b"]
    assert [s.name for s in profile.confident(Confidence.LIKELY)] == ["a"]


def test_unknown_descriptor_is_a_key_error() -> None:
    with pytest.raises(KeyError):
        OdorProfile.from_scores([score("a", 0.5)])["b"]


# --- names ---------------------------------------------------------------------------

VANILLIN = MoleculeNames(cid=1183, en="Vanillin", ru="ванилин")


class StubDirectory:
    def find(self, text: str) -> tuple[str, MoleculeNames, str] | None:
        if text.casefold() in {"ванилин", "vanillin"}:
            return "CCO", VANILLIN, "ванилин"
        return None

    def names_of(self, smiles: str) -> MoleculeNames | None:
        return VANILLIN if smiles == "CCO" else None

    def suggest(self, text: str, limit: int) -> tuple[NameSuggestion, ...]:
        return ()


@dataclass
class StubResolver:
    answer: tuple[str, MoleculeNames] | None = None
    down: bool = False
    calls: list[str] = field(default_factory=list)

    def resolve(self, name: str) -> tuple[str, MoleculeNames] | None:
        self.calls.append(name)
        if self.down:
            raise NameLookupUnavailable("PubChem could not be reached")
        return self.answer


def predict_with(resolver: StubResolver | None = None) -> PredictOdor:
    return PredictOdor(
        reader=StubReader(),
        scope=StubScope(refuse={"CCBr"}),
        predictor=StubPredictor(),
        references=StubReferences(),
        names=StubDirectory(),
        resolver=resolver,
    )


def test_a_known_name_is_answered_from_the_directory_without_going_online() -> None:
    resolver = StubResolver()
    report = predict_with(resolver)("ванилин")

    assert isinstance(report, OdorReport)
    assert report.query_kind == "name"
    assert report.names == VANILLIN
    assert report.matched_name == "ванилин"
    assert resolver.calls == []


def test_a_smiles_input_still_gets_its_name() -> None:
    report = predict_with()("CCO")

    assert isinstance(report, OdorReport)
    assert report.query_kind == "smiles"
    assert report.names == VANILLIN


def test_an_unknown_name_goes_online_and_prefers_the_local_names() -> None:
    online = MoleculeNames(cid=1183, en="Vanillin", ru=None)
    resolver = StubResolver(answer=("CCO", online))

    report = predict_with(resolver)("galaxolide")

    assert isinstance(report, OdorReport)
    assert resolver.calls == ["galaxolide"]
    assert report.names == VANILLIN  # the local directory knows the Russian name


def test_a_name_nobody_knows_is_refused_as_unknown() -> None:
    result = predict_with(StubResolver(answer=None))("galaxolide")

    assert isinstance(result, Refusal)
    assert result.reason is RefusalReason.UNKNOWN_NAME


def test_an_unreachable_name_service_is_not_reported_as_an_unknown_name() -> None:
    """ "PubChem is down" and "no such molecule" are different answers to the user."""
    result = predict_with(StubResolver(down=True))("galaxolide")

    assert isinstance(result, Refusal)
    assert result.reason is RefusalReason.NAME_LOOKUP_UNAVAILABLE


def test_broken_smiles_never_reaches_the_name_service() -> None:
    resolver = StubResolver()
    result = predict_with(resolver)("C1(C")

    assert isinstance(result, Refusal)
    assert result.reason is RefusalReason.INVALID_INPUT
    assert resolver.calls == []


def test_neighbours_without_a_known_name_stay_unnamed() -> None:
    report = predict_with()("CCO", neighbours=1)

    assert isinstance(report, OdorReport)
    assert report.neighbours[0].names is None
