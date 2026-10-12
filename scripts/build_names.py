"""Build the name directory for the reference molecules: English and Russian names.

    uv run python scripts/build_names.py          # ~2 minutes, writes models/<name>/names.json

The dataset carries structures only (`nonStereoSMILES`), no names and no PubChem ids.
Names are looked up once, offline, and shipped next to the model so that the service
answers "ванилин" instantly and without depending on anyone's uptime:

1. **PubChem**, by InChIKey computed from each structure: the compound id, its title
   ("Vanillin") and a few name-like English synonyms ("4-hydroxy-3-methoxybenzaldehyde").
   InChIKeys go in batches; PubChem returns each match with its key, so results map
   back to molecules unambiguously.
2. **PubChem stereoisomers** of the same structure. The dataset has no stereo, so it
   matches PubChem's "mixture of isomers" record, whose names are often systematic
   ("3-Phenylprop-2-enal"); the familiar name sits on the record of one isomer
   ("Cinnamaldehyde" is the E form, "Geraniol" and "Nerol" are E and Z of one
   structure). The first 14 characters of an InChIKey encode the skeleton without
   stereo, so isomer names are grouped back by that block. Both "geraniol" and "nerol"
   then lead to the same molecule -- which is exactly what the model can tell apart.
3. **Wikidata**, by PubChem id: the Russian label ("ванилин") and Russian aliases.

PubChem data is public domain, Wikidata is CC0. Both are credited in the model card.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator, Sequence

from rdkit import Chem, RDLogger

from sillage.paths import DEFAULT_MODEL, project_root

RDLogger.DisableLog("rdApp.*")

PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
WIKIDATA = "https://query.wikidata.org/sparql"
USER_AGENT = "sillage/0.1 (https://github.com/ykvnkm/sillage; odour research project)"
PUBCHEM_PAUSE = 0.25  # PubChem asks for at most 5 requests per second
MAX_ALIASES = 15

# A synonym worth searching by: letters first, ordinary punctuation, short. This drops
# registry numbers ("121-33-5"), vendor codes ("FEMA 3107", "NSC-15351"), and
# database identifiers that make up most of a PubChem synonym list.
NAME_LIKE = re.compile(r"^[A-Za-z][A-Za-z0-9 ,'()\[\]+.\-]{2,40}$")
CODE_LIKE = re.compile(
    r"^(FEMA|NSC|CAS|EINECS|EC |UNII|CHEBI|CHEMBL|DTXSID|DTXCID|SCHEMBL|AKOS|HSDB|BRN|AI3|"
    r"MFCD|NS\d|ZINC|BDBM|HY-|CS-|SY\d|BIDD|LS-|Q\d)",
    re.I,
)
INCHIKEY_LIKE = re.compile(r"^[A-Z]{14}-[A-Z]{8,10}")
VENDOR_CODE = re.compile(r"^[A-Za-z]{1,5}\d{3,}$")  # "orb1222698", "s4970"
ISOTOPIC = re.compile(r"(-d\d+\b|\b13C\b|\(2H\d*\)|deuter|\(13C)", re.I)
MAX_ISOMER_NAMES = 5
# Stereo or mixture suffixes PubChem appends to titles: "Carvone, (+-)-", "Menthol, (-)-".
TITLE_SUFFIX = re.compile(r",\s*\([^)]*\)-?\s*$")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default=DEFAULT_MODEL)
    args = parser.parse_args()

    directory = project_root() / "models" / args.name
    reference = json.loads((directory / "reference.json").read_text("utf-8"))
    smiles = [entry["smiles"] for entry in reference]

    key_of = {s: inchikey(s) for s in smiles}
    found = pubchem_by_inchikey(sorted({k for k in key_of.values() if k}))
    print(f"pubchem: {len(found)} of {len(set(key_of.values()))} structures found", file=sys.stderr)

    cids = sorted({row["cid"] for row in found.values()})
    synonyms = pubchem_synonyms(cids)
    isomer_names = pubchem_isomer_titles(cids)
    print(f"pubchem: names of stereoisomers for {len(isomer_names)} skeletons", file=sys.stderr)
    russian = wikidata_russian(cids)
    print(f"wikidata: {len(russian)} of {len(cids)} compounds have a Russian name", file=sys.stderr)

    names: dict[str, dict] = {}
    for s in smiles:
        row = found.get(key_of[s])
        if row is None:
            continue
        cid = row["cid"]
        ru = russian.get(cid, {})
        names[s] = {
            "cid": cid,
            "en": row["title"],
            "ru": ru.get("label"),
            "aliases": unique(
                [
                    *isomer_names.get(key_of[s][:14], []),
                    *synonyms.get(cid, []),
                    *ru.get("aliases", []),
                ],
                exclude={row["title"], ru.get("label")},
            )[:MAX_ALIASES],
        }

    out = directory / "names.json"
    out.write_text(json.dumps(names, ensure_ascii=False, indent=0) + "\n", "utf-8")
    with_ru = sum(1 for n in names.values() if n["ru"])
    print(
        f"wrote {out.relative_to(project_root())}: {len(names)} of {len(smiles)} molecules named, "
        f"{with_ru} with a Russian name"
    )
    return 0


def inchikey(smiles: str) -> str | None:
    molecule = Chem.MolFromSmiles(smiles)
    return Chem.MolToInchiKey(molecule) if molecule is not None else None


def pubchem_by_inchikey(keys: Sequence[str]) -> dict[str, dict]:
    """InChIKey -> {cid, title}. Lowest id wins when one key matches several records."""
    found: dict[str, dict] = {}
    for batch in chunks(keys, 100):
        body = urllib.parse.urlencode({"inchikey": ",".join(batch)}).encode()
        payload = request(f"{PUBCHEM}/compound/inchikey/property/InChIKey,Title/JSON", body)
        for row in (payload or {}).get("PropertyTable", {}).get("Properties", []):
            key, cid = row.get("InChIKey"), row["CID"]
            if key and row.get("Title") and (key not in found or cid < found[key]["cid"]):
                found[key] = {"cid": cid, "title": TITLE_SUFFIX.sub("", row["Title"])}
        time.sleep(PUBCHEM_PAUSE)
    return found


def pubchem_synonyms(cids: Sequence[int]) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    for batch in chunks(cids, 100):
        body = urllib.parse.urlencode({"cid": ",".join(map(str, batch))}).encode()
        payload = request(f"{PUBCHEM}/compound/cid/synonyms/JSON", body)
        for info in (payload or {}).get("InformationList", {}).get("Information", []):
            names = [s for s in info.get("Synonym", []) if is_name(s)]
            out[info["CID"]] = names[: MAX_ALIASES * 2]
        time.sleep(PUBCHEM_PAUSE)
    return out


def pubchem_isomer_titles(cids: Sequence[int]) -> dict[str, list[str]]:
    """InChIKey skeleton block -> titles of every record sharing that skeleton."""
    variants: set[int] = set()
    for batch in chunks(cids, 100):
        body = urllib.parse.urlencode({"cid": ",".join(map(str, batch))}).encode()
        url = f"{PUBCHEM}/compound/cid/cids/JSON?cids_type=same_connectivity"
        payload = request(url, body)
        variants.update((payload or {}).get("IdentifierList", {}).get("CID", []))
        time.sleep(PUBCHEM_PAUSE)

    titles: dict[str, list[str]] = {}
    for batch in chunks(sorted(variants - set(cids)), 300):
        body = urllib.parse.urlencode({"cid": ",".join(map(str, batch))}).encode()
        payload = request(f"{PUBCHEM}/compound/cid/property/InChIKey,Title/JSON", body)
        for row in (payload or {}).get("PropertyTable", {}).get("Properties", []):
            title, key = row.get("Title"), row.get("InChIKey")
            if title and key and is_name(title) and not ISOTOPIC.search(title):
                titles.setdefault(key[:14], []).append(TITLE_SUFFIX.sub("", title))
        time.sleep(PUBCHEM_PAUSE)
    # Short names first: "Geraniol" before "(2E)-3,7-dimethylocta-2,6-dien-1-ol".
    return {
        block: sorted(set(names), key=len)[:MAX_ISOMER_NAMES] for block, names in titles.items()
    }


def wikidata_russian(cids: Sequence[int]) -> dict[int, dict]:
    """PubChem id -> {label, aliases} in Russian, from Wikidata's P662 property."""
    out: dict[int, dict] = {}
    for batch in chunks(cids, 300):
        values = " ".join(f'"{cid}"' for cid in batch)
        query = f"""
            SELECT ?cid ?label (GROUP_CONCAT(DISTINCT ?alias; separator="|") AS ?aliases) WHERE {{
              VALUES ?cid {{ {values} }}
              ?item wdt:P662 ?cid .
              ?item rdfs:label ?label . FILTER(lang(?label) = "ru")
              OPTIONAL {{ ?item skos:altLabel ?alias . FILTER(lang(?alias) = "ru") }}
            }} GROUP BY ?cid ?label"""
        url = f"{WIKIDATA}?{urllib.parse.urlencode({'query': query})}"
        payload = request(url, accept="application/sparql-results+json")
        for row in (payload or {}).get("results", {}).get("bindings", []):
            cid = int(row["cid"]["value"])
            aliases = [a for a in row.get("aliases", {}).get("value", "").split("|") if a]
            out.setdefault(cid, {"label": row["label"]["value"], "aliases": aliases})
        time.sleep(1.0)
    return out


def request(url: str, body: bytes | None = None, accept: str = "application/json") -> dict | None:
    headers = {"User-Agent": USER_AGENT, "Accept": accept}
    for attempt in range(1, 5):
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, data=body, headers=headers), timeout=60
            ) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            if error.code == 404:  # none of the batch matched
                return None
            if attempt == 4:
                raise
        except OSError:
            if attempt == 4:
                raise
        time.sleep(2**attempt)
    return None


def chunks(items: Sequence, size: int) -> Iterator[Sequence]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def is_name(text: str) -> bool:
    """A synonym a person might type, not a registry number or database code."""
    if not NAME_LIKE.match(text) or CODE_LIKE.match(text) or INCHIKEY_LIKE.match(text):
        return False
    if VENDOR_CODE.match(text):
        return False
    # Upper-case tokens with digits are identifiers ("X796XFP7D4"), not names.
    return any(c.islower() for c in text) or not any(c.isdigit() for c in text)


def unique(items: list[str | None], exclude: set[str | None]) -> list[str]:
    seen = {e.casefold() for e in exclude if e}
    out = []
    for item in items:
        if item and item.casefold() not in seen:
            seen.add(item.casefold())
            out.append(item)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
