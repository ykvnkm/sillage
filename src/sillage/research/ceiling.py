"""How far do the experts agree with each other? The ceiling for any model of their labels.

The training labels are the union of two independent expert collections, Goodscents
and Leffingwell, merged by OpenPOM. Where both described the same molecule, their
disagreement is a property of the data, not of any model: no model can agree with
one collection more reliably than the other collection does.

Agreement is Cohen's kappa per descriptor -- 0 is chance, 1 is identity -- the usual
measure of inter-rater reliability. The model is put on the same scale by marking a
descriptor present when it calls it more likely than not (probability >= 0.5), and is
compared with each collection separately, on molecules it saw neither in training nor
in calibration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from sklearn.metrics import cohen_kappa_score

from sillage.data.schemas import OPENPOM_DESCRIPTORS
from sillage.paths import raw_data_dir

RDLogger.DisableLog("rdApp.*")

MIN_POSITIVES = 5
"""Per collection, on the evaluated molecules. Kappa on two or three positives is a
coin toss; such descriptors are reported but left out of averages."""


@dataclass(frozen=True, slots=True)
class Collections:
    """Both collections on the shared vocabulary, keyed by stereo-free SMILES."""

    goodscents: dict[str, frozenset[str]]
    leffingwell: dict[str, frozenset[str]]
    vocabulary: tuple[str, ...]
    """Descriptors that exist in our 138 *and* in both collections' vocabularies."""

    @property
    def shared_molecules(self) -> list[str]:
        return sorted(set(self.goodscents) & set(self.leffingwell))


def load() -> Collections:
    raw = raw_data_dir()

    lf_molecules = pd.read_csv(raw / "pyrfume_leffingwell_molecules.csv")
    lf_behaviour = pd.read_csv(raw / "pyrfume_leffingwell_behavior.csv")
    smiles_of = dict(zip(lf_molecules["CID"], lf_molecules["IsomericSMILES"], strict=True))
    lf_vocabulary = [c for c in lf_behaviour.columns if c != "Stimulus"]
    leffingwell: dict[str, set[str]] = {}
    for _, row in lf_behaviour.iterrows():
        key = _flat(smiles_of.get(row["Stimulus"]))
        if key:
            leffingwell.setdefault(key, set()).update(c for c in lf_vocabulary if row[c] == 1)

    gs_molecules = pd.read_csv(raw / "pyrfume_goodscents_molecules.csv")
    gs_behaviour = pd.read_csv(raw / "pyrfume_goodscents_behavior.csv")
    gs_stimuli = pd.read_csv(raw / "pyrfume_goodscents_stimuli.csv")
    smiles_of = dict(zip(gs_molecules["CID"], gs_molecules["IsomericSMILES"], strict=True))
    descriptors = zip(gs_behaviour["Stimulus"], gs_behaviour["Descriptors"], strict=True)
    phrases_of = {
        stimulus: {p.strip() for p in str(text).split(";") if p.strip()}
        for stimulus, text in descriptors
    }
    goodscents: dict[str, set[str]] = {}
    gs_vocabulary: set[str] = set()
    for stimulus, cid in zip(gs_stimuli["Stimulus"], gs_stimuli["CID"], strict=True):
        key = _flat(smiles_of.get(cid))
        if key and stimulus in phrases_of:
            goodscents.setdefault(key, set()).update(phrases_of[stimulus])
            gs_vocabulary |= phrases_of[stimulus]

    vocabulary = tuple(d for d in OPENPOM_DESCRIPTORS if d in lf_vocabulary and d in gs_vocabulary)
    keep = set(vocabulary)
    return Collections(
        goodscents={k: frozenset(v & keep) for k, v in goodscents.items()},
        leffingwell={k: frozenset(v & keep) for k, v in leffingwell.items()},
        vocabulary=vocabulary,
    )


def matrix(labels: dict[str, frozenset[str]], molecules: list[str], vocabulary) -> np.ndarray:
    return np.array([[d in labels[m] for d in vocabulary] for m in molecules], dtype=np.int8)


def kappas(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cohen's kappa per column; NaN where either side never says yes (undefined)."""
    out = np.full(a.shape[1], np.nan)
    for j in range(a.shape[1]):
        if a[:, j].any() and b[:, j].any():
            out[j] = cohen_kappa_score(a[:, j], b[:, j])
    return out


def _flat(smiles: object) -> str | None:
    if not isinstance(smiles, str):
        return None
    molecule = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(molecule, isomericSmiles=False) if molecule is not None else None
