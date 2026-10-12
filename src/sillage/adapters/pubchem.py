"""Names the local directory does not know, resolved through PubChem.

The local directory covers the 4733 named reference molecules. Anything else a user
types -- a new aroma chemical, a common solvent -- goes to PubChem's PUG REST API,
which answers with a structure, stereochemistry included.

A short timeout and a cache, because this sits on the request path of a public
page: a slow PubChem must not make the whole page slow, and the same name asked
twice must not cost two round trips.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from functools import lru_cache

from sillage.domain import MoleculeNames
from sillage.domain.ports import NameLookupUnavailable

PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
USER_AGENT = "sillage/0.1 (https://github.com/ykvnkm/sillage)"
TIMEOUT_SECONDS = 4.0


class PubChemResolver:
    def __init__(self, timeout: float = TIMEOUT_SECONDS) -> None:
        self._timeout = timeout
        self._lookup = lru_cache(maxsize=1024)(self._fetch)

    def resolve(self, name: str) -> tuple[str, MoleculeNames] | None:
        return self._lookup(name.strip().casefold())

    def _fetch(self, name: str) -> tuple[str, MoleculeNames] | None:
        url = f"{PUG}/compound/name/{urllib.parse.quote(name, safe='')}/property/SMILES,Title/JSON"
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            if error.code == 404:  # PubChem's answer for "no such name"
                return None
            raise NameLookupUnavailable(f"PubChem answered {error.code}") from error
        except (OSError, ValueError) as error:
            raise NameLookupUnavailable(f"PubChem could not be reached: {error}") from error

        rows = payload.get("PropertyTable", {}).get("Properties", [])
        if not rows or not rows[0].get("SMILES"):
            return None
        row = rows[0]  # PubChem orders matches by relevance
        return row["SMILES"], MoleculeNames(cid=row.get("CID"), en=row.get("Title"), ru=None)
