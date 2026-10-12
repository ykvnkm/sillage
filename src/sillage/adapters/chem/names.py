"""The name directory: every reference molecule's names, matched by folded spelling."""

from __future__ import annotations

import json
import re
from bisect import bisect_left
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

from sillage.domain import MoleculeNames, NameSuggestion, fold, initial

NAMES_FILE = "names.json"

# How a matched name ranks: what the molecule is called beats one of its aliases.
_PRIMARY, _ALIAS = 0, 1


class NameIndex:
    """``NameDirectory`` over ``names.json``, built by ``scripts/build_names.py``."""

    def __init__(self, names: Mapping[str, MoleculeNames]) -> None:
        self._names = dict(names)
        # folded key -> [(rank, length of the name, smiles, the name as written)]
        self._keys: dict[str, list[tuple[int, int, str, str]]] = defaultdict(list)
        for smiles, entry in self._names.items():
            primary = {entry.ru, entry.en}
            # When the molecule's own name is systematic, its aliases are where the
            # names people use live ("Geraniol" for 3,7-Dimethyl-2,6-octadienol).
            aliases_are_common = _systematic(entry.display or "")
            for name in entry.all():
                key = fold(name)
                if key:
                    common = name in primary or aliases_are_common
                    rank = _PRIMARY if common else _ALIAS
                    self._keys[key].append((rank, len(name), smiles, name))
        for matches in self._keys.values():
            matches.sort()
        self._sorted_keys = sorted(self._keys)

    @classmethod
    def from_file(cls, path: Path) -> NameIndex:
        raw = json.loads(Path(path).read_text("utf-8"))
        return cls(
            {
                smiles: MoleculeNames(
                    cid=entry.get("cid"),
                    en=entry.get("en"),
                    ru=entry.get("ru"),
                    aliases=tuple(entry.get("aliases", ())),
                )
                for smiles, entry in raw.items()
            }
        )

    def __len__(self) -> int:
        return len(self._names)

    def find(self, text: str) -> tuple[str, MoleculeNames, str] | None:
        matches = self._keys.get(fold(text))
        if not matches:
            return None
        _, _, smiles, name = matches[0]
        return smiles, self._names[smiles], name

    def names_of(self, smiles: str) -> MoleculeNames | None:
        return self._names.get(smiles)

    def suggest(self, text: str, limit: int) -> tuple[NameSuggestion, ...]:
        query = fold(text)
        if len(query) < 2 or limit <= 0:
            return ()

        # Exact matches first; then names starting with the letter actually typed
        # ("гер" means Geraniol before Herbac, which matches only through h -> g);
        # then a molecule's common name before a trade synonym ("ван" is vanillin,
        # not triacetin sold as "Vanay"); then the shortest name.
        typed = initial(text)
        candidates: list[tuple[tuple[int, int, int, int], str, str]] = []
        for key in self._prefixed(query):
            for rank, length, smiles, name in self._keys[key]:
                exact = 0 if key == query else 1
                other_letter = 0 if initial(name) == typed else 1
                candidates.append(((exact, other_letter, rank, length), smiles, name))
        candidates.sort()

        seen: set[str] = set()
        out: list[NameSuggestion] = []
        for _, smiles, name in candidates:
            if smiles in seen:
                continue
            seen.add(smiles)
            display = self._names[smiles].display or name
            if _systematic(display):
                display = name  # "Geraniol", not "3,7-Dimethyl-2,6-octadienol"
            out.append(NameSuggestion(display=display, matched=name, smiles=smiles))
            if len(out) == limit:
                break
        return tuple(out)

    def _prefixed(self, query: str) -> list[str]:
        """Folded keys starting with *query*, found by binary search on the sorted keys."""
        start = bisect_left(self._sorted_keys, query)
        keys = []
        for key in self._sorted_keys[start:]:
            if not key.startswith(query):
                break
            keys.append(key)
        return keys


def _systematic(name: str) -> bool:
    """A systematic chemical name, correct but not what anyone types: locants and commas."""
    return bool(re.search(r"\d|,", name)) or len(name) > 30
