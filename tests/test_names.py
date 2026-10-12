"""Name matching: Russian transliterations meet English names; PubChem is optional.

Only 388 of 4733 named reference molecules have a Russian name in Wikidata. The rest
are reachable from Russian because chemical names transliterate, and :func:`fold`
reduces both spellings to the same skeleton. These pairs are the contract.
"""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

from sillage.adapters import pubchem
from sillage.adapters.chem.names import NameIndex
from sillage.adapters.pubchem import PubChemResolver
from sillage.domain import MoleculeNames, fold, looks_like_name
from sillage.domain.ports import NameLookupUnavailable


@pytest.mark.parametrize(
    ("english", "russian"),
    [
        ("vanillin", "ванилин"),
        ("hedione", "гедион"),
        ("ambroxide", "амброксид"),
        ("Ethyl butyrate", "этилбутират"),
        ("linalool", "линалоол"),
        ("citral", "цитраль"),
        ("eugenol", "эвгенол"),
        ("coumarin", "кумарин"),
        ("cis-3-Hexenol", "цис-3-гексенол"),
        ("jasmone", "жасмон"),
        ("camphor", "камфора"),
        ("Iso E Super", "изо е супер"),
    ],
)
def test_transliterations_fold_together(english: str, russian: str) -> None:
    assert fold(english) == fold(russian)


@pytest.mark.parametrize(("a", "b"), [("vanillin", "linalool"), ("menthol", "methanol")])
def test_different_names_stay_different(a: str, b: str) -> None:
    assert fold(a) != fold(b)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ванилин", True),
        ("hedione", True),
        ("Ethyl 2-methylbutyrate", True),
        ("COc1cc(C=O)ccc1O", False),
        ("C[C@H](O)CC", False),
        ("CCO", False),
    ],
)
def test_names_are_told_from_smiles(text: str, expected: bool) -> None:
    assert looks_like_name(text) is expected


# --- the directory ----------------------------------------------------------------------


@pytest.fixture
def index() -> NameIndex:
    return NameIndex(
        {
            "CCO": MoleculeNames(cid=702, en="Ethanol", ru="этанол", aliases=("Ethyl alcohol",)),
            "CC(C)O": MoleculeNames(cid=3776, en="Isopropyl alcohol", ru=None, aliases=()),
            "CCCO": MoleculeNames(cid=1031, en="1-Propanol", ru=None, aliases=("Ethylcarbinol",)),
        }
    )


def test_find_is_exact_after_folding(index: NameIndex) -> None:
    assert index.find("ЭТАНОЛ")[0] == "CCO"
    assert index.find("ethyl-alcohol")[0] == "CCO"
    assert index.find("этан") is None  # a prefix is a suggestion, not a match


def test_suggest_ranks_by_length_and_shows_the_name_that_matched(index: NameIndex) -> None:
    names = [s.display for s in index.suggest("eth", limit=5)]

    # Ethanol matches by its own name; 1-propanol only through the alias Ethylcarbinol,
    # which is shown instead of the systematic "1-Propanol" the user did not type.
    assert names == ["этанол", "Ethylcarbinol"]


def test_suggest_lists_each_molecule_once(index: NameIndex) -> None:
    smiles = [s.smiles for s in index.suggest("et", limit=10)]

    assert len(smiles) == len(set(smiles))


def test_suggest_ignores_a_single_letter(index: NameIndex) -> None:
    assert index.suggest("e", limit=5) == ()


# --- PubChem --------------------------------------------------------------------------


class _Response(io.BytesIO):
    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def test_pubchem_answer_becomes_a_structure_and_names(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "PropertyTable": {
            "Properties": [{"CID": 6549, "SMILES": "CC(=CCCC(C)(C=C)O)C", "Title": "Linalool"}]
        }
    }
    seen: list[str] = []

    def fake_urlopen(request, timeout):
        seen.append(request.full_url)
        return _Response(json.dumps(payload).encode())

    monkeypatch.setattr(pubchem.urllib.request, "urlopen", fake_urlopen)
    resolver = PubChemResolver()

    smiles, names = resolver.resolve("Linalool")
    resolver.resolve("linalool")  # same name, different case: cached

    assert smiles == "CC(=CCCC(C)(C=C)O)C"
    assert names == MoleculeNames(cid=6549, en="Linalool", ru=None)
    assert len(seen) == 1
    assert "/compound/name/linalool/property/SMILES,Title/JSON" in seen[0]


def test_pubchem_404_means_no_such_name(monkeypatch: pytest.MonkeyPatch) -> None:
    def not_found(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 404, "Not Found", None, None)

    monkeypatch.setattr(pubchem.urllib.request, "urlopen", not_found)

    assert PubChemResolver().resolve("no such molecule") is None


@pytest.mark.parametrize(
    "failure",
    [
        urllib.error.HTTPError("u", 503, "Unavailable", None, None),
        urllib.error.URLError("timed out"),
    ],
)
def test_pubchem_failure_is_unavailable_not_unknown(
    monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    def broken(request, timeout):
        raise failure

    monkeypatch.setattr(pubchem.urllib.request, "urlopen", broken)

    with pytest.raises(NameLookupUnavailable):
        PubChemResolver().resolve("linalool")


def test_suggest_prefers_a_familiar_name_over_a_systematic_one() -> None:
    index = NameIndex(
        {
            "CC(C)=CCCC(C)=CCO": MoleculeNames(
                cid=4458, en="3,7-Dimethyl-2,6-octadienol", ru=None, aliases=("Geraniol", "Nerol")
            ),
            "CC(C)=CCCC(C)=CCOC(=O)CC(C)CC": MoleculeNames(
                cid=1, en="Geranyl 2-methyl butyrate", ru=None
            ),
        }
    )

    first = index.suggest("гер", limit=5)[0]

    assert first.display == "Geraniol"
    assert first.smiles == "CC(C)=CCCC(C)=CCO"


def test_name_builder_keeps_names_and_drops_codes() -> None:
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "build_names.py"
    spec = importlib.util.spec_from_file_location("build_names", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for code in [
        "DTXSID7025502",
        "MFCD00008377",
        "UJCFZCTTZWHRNL-UHFFFAOYSA-N",
        "X796XFP7D4",
        "orb1222698",
        "s4970",
        "SCHEMBL2865536",
    ]:
        assert not module.is_name(code), code
    for name in ["Iso-E super", "AMBROX DL", "Methyl dihydrojasmonate", "Linalol"]:
        assert module.is_name(name), name


def test_the_typed_first_letter_outranks_a_transliteration_fold() -> None:
    index = NameIndex(
        {
            "A": MoleculeNames(cid=1, en="Herbac", ru=None),
            "B": MoleculeNames(cid=2, en="Geraniol", ru=None),
        }
    )

    assert [s.display for s in index.suggest("гер", limit=2)] == ["Geraniol", "Herbac"]
    assert [s.display for s in index.suggest("her", limit=2)] == ["Herbac", "Geraniol"]


def test_a_common_name_outranks_a_shorter_trade_synonym() -> None:
    index = NameIndex(
        {
            "VAN": MoleculeNames(cid=1183, en="Vanillin", ru="ванилин"),
            "TRI": MoleculeNames(cid=5541, en="Triacetin", ru="триацетин", aliases=("Vanay",)),
        }
    )

    assert [s.display for s in index.suggest("ван", limit=2)] == ["ванилин", "триацетин"]
