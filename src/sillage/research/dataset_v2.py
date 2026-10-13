"""Dataset v2: the two expert collections rebuilt from their own records, with three fixes.

The training data so far is OpenPOM's merge of Goodscents and Leffingwell: one SMILES
without stereochemistry, and a label wherever either collection used the word. Two
measurements showed what that costs:

* **"Not asked" became "no".** Leffingwell's vocabulary has no `jasmin`, `lily`,
  `amber` and ~30 more. A molecule only Leffingwell described can never carry them,
  so the merge teaches the model it is not jasmine-like. Here a label a collection
  cannot express is *missing* for that collection, not negative.
* **Disagreement became certainty.** Experts agree on kappa 0.48 (E-018); where one
  collection says "warm" and the other does not, the merge says "warm". Here the
  target is the share of collections that said yes: 1, 0.5 or 0.
* **Stereochemistry was dropped** for 54% of molecules (E-015). Here every record
  keeps its isomeric structure; isomers of one skeleton share a split group so that
  (R)- in training and (S)- in test cannot leak into each other.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Mol

from sillage.data.schemas import OPENPOM_DESCRIPTORS
from sillage.models import ChemicalScope, element_vocabulary
from sillage.paths import raw_data_dir
from sillage.splits import Split, iterative_stratification

RDLogger.DisableLog("rdApp.*")

LABELS: tuple[str, ...] = tuple(OPENPOM_DESCRIPTORS)


@dataclass(frozen=True, slots=True)
class DatasetV2:
    smiles: tuple[str, ...]
    """Canonical isomeric SMILES: stereo kept where the record had it."""

    groups: tuple[str, ...]
    """Stereo-free SMILES: isomers of one skeleton share a group."""

    molecules: list[Mol]
    goodscents: np.ndarray
    """Votes, shape (n, 138): 1 said, 0 could have said and did not, NaN not asked."""

    leffingwell: np.ndarray

    scope: ChemicalScope | None = None
    """The declared scope the rows were filtered to, element vocabulary included."""

    @property
    def label_names(self) -> tuple[str, ...]:
        return LABELS

    def targets(self, *, soft: bool = True, mask_unasked: bool = True) -> np.ndarray:
        """The training target under a given policy.

        ``soft`` -- disagreement is 0.5 (else: any yes is 1, the merge's rule).
        ``mask_unasked`` -- a label no collection could express is NaN (else: 0).
        """
        votes = np.stack([self.goodscents, self.leffingwell])
        asked = ~np.isnan(votes)
        if soft:
            # Cells nobody was asked about are all-NaN; NaN is the intended result.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                target = np.nanmean(votes, axis=0)
        else:
            any_yes = np.nan_to_num(votes, nan=0.0).max(axis=0)
            target = np.where(asked.any(axis=0), any_yes, np.nan)
        if not mask_unasked:
            target = np.nan_to_num(target, nan=0.0)
        return target

    def consensus(self) -> np.ndarray:
        """Labels both collections were asked about and agree on; NaN elsewhere.

        Only molecules described by both carry any. Scoring against these measures the
        model's skill without the annotators' disagreement mixed in.
        """
        agree = (self.goodscents == self.leffingwell) & ~np.isnan(self.goodscents)
        return np.where(agree, self.goodscents, np.nan)

    def restrict(self, keep: np.ndarray) -> DatasetV2:
        keep = np.asarray(keep, dtype=bool)
        return DatasetV2(
            smiles=tuple(s for s, k in zip(self.smiles, keep, strict=True) if k),
            groups=tuple(g for g, k in zip(self.groups, keep, strict=True) if k),
            molecules=[m for m, k in zip(self.molecules, keep, strict=True) if k],
            goodscents=self.goodscents[keep],
            leffingwell=self.leffingwell[keep],
            scope=self.scope,
        )

    def split(self, seed: int = 0) -> Split:
        """Iterative stratification over skeleton groups, mapped back to rows."""
        group_names = sorted(set(self.groups))
        index = {g: i for i, g in enumerate(group_names)}
        member = np.array([index[g] for g in self.groups])
        positive = np.nan_to_num(self.targets(soft=False), nan=0.0) >= 0.5
        group_labels = np.zeros((len(group_names), len(LABELS)), dtype=np.int8)
        np.maximum.at(group_labels, member, positive.astype(np.int8))
        by_group = iterative_stratification(group_labels, seed=seed)
        parts = {}
        for part, groups in by_group.parts().items():
            parts[part] = np.flatnonzero(np.isin(member, groups))
        return Split(**parts, strategy="stratified-by-skeleton", seed=seed)


def build(scope: ChemicalScope | None = None) -> DatasetV2:
    """Both collections, joined by isomeric structure, filtered to the model's scope."""
    gs = _goodscents()
    lf = _leffingwell()
    keys = sorted(set(gs.votes) | set(lf.votes))

    def stack(collection: _Collection) -> np.ndarray:
        out = np.full((len(keys), len(LABELS)), np.nan)
        for i, key in enumerate(keys):
            said = collection.votes.get(key)
            if said is not None:
                out[i, collection.expressible] = 0.0
                out[i, [LABELS.index(d) for d in said]] = 1.0
        return out

    molecules = [Chem.MolFromSmiles(k) for k in keys]
    data = DatasetV2(
        smiles=tuple(keys),
        groups=tuple(Chem.MolToSmiles(m, isomericSmiles=False) for m in molecules),
        molecules=molecules,
        goodscents=stack(gs),
        leffingwell=stack(lf),
    )

    scope = scope or ChemicalScope()
    data = data.restrict(scope.contains(data.molecules))
    vocabulary = element_vocabulary(data.molecules)
    scope = ChemicalScope(**{**_fields(scope), "allowed_elements": vocabulary})
    data = data.restrict(scope.contains(data.molecules))
    return replace(data, scope=scope)


@dataclass(frozen=True, slots=True)
class _Collection:
    votes: dict[str, set[str]]
    expressible: list[int]
    """Indices into LABELS of descriptors this collection's vocabulary contains."""


# Goodscents writes descriptors as free phrases: "minty", "almond bitter almond",
# "onion cooked onion", "lily of the valley". OpenPOM read them word by word; exact
# phrase matching alone loses about a thousand positives (journal E-019). A phrase
# therefore contributes every vocabulary word it contains, its adjective forms
# ("minty" -> mint) and the few synonyms below, which name the same note.
_SYNONYMS: dict[str, str] = {
    "spearmint": "mint",
    "peppermint": "mint",
    "cornmint": "mint",
    "mentholic": "cooling",
    "lily of the valley": "muguet",
    "lily-of-the-valley": "muguet",
    "almost odorless": "odorless",
    "ambergris": "amber",
    "neroli": "orangeflower",
    "orange flower": "orangeflower",
    "black currant bud": "black currant",
    "currant bud black currant bud": "black currant",
    "anise": "anisic",
    "licorice": "anisic",
    "horseradish": "radish",
}


def phrase_labels(phrase: str) -> set[str]:
    """Vocabulary descriptors a Goodscents phrase expresses."""
    phrase = phrase.strip().lower()
    found = {d for d in LABELS if d == phrase}
    found |= {_SYNONYMS[phrase]} if phrase in _SYNONYMS else set()
    found |= {d for d in LABELS if " " in d and d in phrase}  # "black currant", "fruit skin"
    for word in phrase.replace("-", " ").split():
        for form in (word, word.removesuffix("y"), word.removesuffix("ic")):
            if form in LABELS:
                found.add(form)
        if word in _SYNONYMS:
            found.add(_SYNONYMS[word])
    return found


def _goodscents() -> _Collection:
    raw = raw_data_dir()
    molecules = pd.read_csv(raw / "pyrfume_goodscents_molecules.csv")
    behaviour = pd.read_csv(raw / "pyrfume_goodscents_behavior.csv")
    stimuli = pd.read_csv(raw / "pyrfume_goodscents_stimuli.csv")
    smiles_of = dict(zip(molecules["CID"], molecules["IsomericSMILES"], strict=True))
    phrases = {
        s: {p.strip() for p in str(t).split(";") if p.strip()}
        for s, t in zip(behaviour["Stimulus"], behaviour["Descriptors"], strict=True)
    }
    mapped = {s: set().union(*(phrase_labels(p) for p in ps)) for s, ps in phrases.items()}
    vocabulary = set().union(*mapped.values())
    votes: dict[str, set[str]] = {}
    for stimulus, cid in zip(stimuli["Stimulus"], stimuli["CID"], strict=True):
        key = _canonical(smiles_of.get(cid))
        if key and stimulus in mapped:
            votes.setdefault(key, set()).update(mapped[stimulus])
    return _Collection(votes, [i for i, d in enumerate(LABELS) if d in vocabulary])


def _leffingwell() -> _Collection:
    raw = raw_data_dir()
    molecules = pd.read_csv(raw / "pyrfume_leffingwell_molecules.csv")
    behaviour = pd.read_csv(raw / "pyrfume_leffingwell_behavior.csv")
    smiles_of = dict(zip(molecules["CID"], molecules["IsomericSMILES"], strict=True))
    columns = [c for c in behaviour.columns if c != "Stimulus"]
    votes: dict[str, set[str]] = {}
    for _, row in behaviour.iterrows():
        key = _canonical(smiles_of.get(row["Stimulus"]))
        if key:
            votes.setdefault(key, set()).update(c for c in columns if row[c] == 1 and c in LABELS)
    return _Collection(votes, [i for i, d in enumerate(LABELS) if d in columns])


def _canonical(smiles: object) -> str | None:
    if not isinstance(smiles, str):
        return None
    molecule = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(molecule) if molecule is not None else None


def _fields(scope: ChemicalScope) -> dict:
    from dataclasses import asdict

    return asdict(scope)
