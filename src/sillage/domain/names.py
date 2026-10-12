"""Molecule names, and how a typed name is matched against them.

Russian names exist for only 388 of the 4733 named reference molecules (Wikidata knows
famous compounds in Russian, not the long tail). But Russian chemical names are, almost
always, transliterations of the English ones: ванилин - vanillin, гедион - hedione,
амброксид - ambroxide, этилбутират - ethyl butyrate. So instead of translating, both
sides are reduced to a common phonetic skeleton and compared there.

:func:`fold` does that reduction. It is lossy on purpose -- "vanilin" and "vanillin"
must collide -- and is only ever used to *match*, never to display.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_CYRILLIC = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e", "ж": "j",
    "з": "z", "и": "i", "й": "i", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o",
    "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "h", "ц": "c",
    "ч": "ch", "ш": "sh", "щ": "sh", "ъ": "", "ы": "i", "ь": "", "э": "e", "ю": "yu",
    "я": "ya",
}  # fmt: skip

# Applied in order, to both the query and every known name. Each line is a spelling
# difference between English chemical names and their Russian transliterations.
_FOLDS = (
    ("ph", "f"),  # phenol - фенол
    ("th", "t"),  # methyl - метил
    ("ch", "h"),  # chlor - хлор
    ("ck", "k"),
    ("x", "ks"),  # ambroxide - амброксид
    ("qu", "kv"),
    ("q", "k"),
    ("w", "v"),
    ("eu", "ev"),  # eugenol - эвгенол
    ("ou", "u"),  # coumarin - кумарин
    ("y", "i"),  # butyrate - бутират
    ("j", "i"),  # jasmone - жасмон
    ("c", "k"),  # citral - цитраль, camphor - камфора
    ("h", "g"),  # hedione - гедион, hexanol - гексанол
    ("z", "s"),  # isoamyl - изоамил
)


def initial(text: str) -> str:
    """First letter after plain transliteration, before any folding: "гер" -> "g".

    Folding maps h to g so that гедион finds hedione; ranking uses this to put a
    typed "гер" next to Geraniol before Herbac, which only matches through that fold.
    """
    for ch in text.casefold():
        latin = _CYRILLIC.get(ch, ch)
        if latin[:1].isalpha():
            return latin[0]
    return ""


def fold(text: str) -> str:
    """Reduce a name to a spelling-insensitive skeleton for matching.

    >>> fold("Ethyl butyrate") == fold("этилбутират")
    True
    """
    folded = "".join(_CYRILLIC.get(ch, ch) for ch in text.casefold())
    folded = re.sub(r"[^a-z0-9]", "", folded)
    for source, target in _FOLDS:
        folded = folded.replace(source, target)
    folded = re.sub(r"(.)\1+", r"\1", folded)  # vanillin - ванилин
    folded = re.sub(r"a$", "", folded)  # camphor/камфора after folding: kamfor/kamfora
    return re.sub(r"e$", "", folded)  # hedione - гедион


def looks_like_name(text: str) -> bool:
    """True for text that could be a name rather than a SMILES string.

    SMILES never contains Cyrillic, and outside brackets it spells atoms with a small
    alphabet (B C N O P S F Cl Br I, lower case when aromatic). A run of letters
    that steps outside it -- "hedione", "CARVONE" -- is a word, while "CCO" is not.
    Names rarely contain the bond and stereo symbols SMILES is made of.
    """
    if re.search(r"[=#@\[\]\\/]", text):
        return False
    if re.search(r"[А-Яа-яЁё]", text):
        return True
    return any(set(run) - set(_SMILES_ATOM_LETTERS) for run in re.findall(r"[A-Za-z]{3,}", text))


_SMILES_ATOM_LETTERS = "BCNOPSFIlrbcnops"


@dataclass(frozen=True, slots=True)
class MoleculeNames:
    """What a molecule is called. Facts from PubChem and Wikidata, not from the model."""

    cid: int | None
    """PubChem compound id; links to the public record."""

    en: str | None
    ru: str | None
    aliases: tuple[str, ...] = ()

    @property
    def display(self) -> str | None:
        """The name to show on a Russian page: Russian when known, English otherwise."""
        return self.ru or self.en

    def all(self) -> tuple[str, ...]:
        return tuple(name for name in (self.ru, self.en, *self.aliases) if name)


@dataclass(frozen=True, slots=True)
class NameSuggestion:
    """One autocomplete entry: the name to show, the name that matched, the structure."""

    display: str
    matched: str
    smiles: str
