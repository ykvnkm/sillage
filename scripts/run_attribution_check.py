"""Does the atom highlighting point where chemistry says it should? (journal E-021)

    uv run python scripts/run_attribution_check.py     # ~2 minutes; results/attribution_check.csv

The pointing game, on test molecules of the shipped model's own split -- molecules
neither its linear and boosting parts nor its graph networks learnt from. For notes
with a textbook carrier group (sulfur for sulfurous, the ester for fruity, ...), over
molecules that experts gave the note and that contain the group: how often is the
atom with the largest contribution inside the group? Chance is the share of the
molecule's atoms the group covers.

Three controls make the number mean something:

* each component on its own, to see which of them points where;
* the graph networks with their weights shuffled -- the sanity check of Adebayo et
  al. (2018): highlighting that survives randomising the model shows the molecule,
  not what the model learnt;
* notes on molecules *without* the group are left out, not counted as misses.
"""

from __future__ import annotations

import warnings
from dataclasses import replace

import numpy as np
import pandas as pd
from rdkit import Chem

from sillage.models.artifact import OdorModel
from sillage.models.attribution import explain
from sillage.models.graph_network import GraphEnsemble, GraphNetwork
from sillage.paths import project_root
from sillage.research.dataset_v2 import build

MODEL = "odor-ensemble-v3"
SULFUR = "[#16]"
CARRIERS = {
    "sulfurous": SULFUR,
    "alliaceous": SULFUR,
    "garlic": SULFUR,
    "onion": SULFUR,
    "cabbage": SULFUR,
    "meaty": SULFUR,
    "fruity": "[CX3](=O)[OX2H0][#6]",  # ester
    "aldehydic": "[CX3H1](=O)[#6]",
    "phenolic": "c[OX2H]",
    "smoky": "c[OX2H]",
    "nutty": "[n]",
    "roasted": "[n]",
    "sour": "[CX3](=O)[OX2H]",  # carboxylic acid
    "cheesy": "[CX3](=O)[OX2H]",
    "sweaty": "[CX3](=O)[OX2H]",
    "fishy": "[NX3;!$(N-C=O);!$(N-a)]",  # aliphatic amine
}


def main() -> int:
    warnings.filterwarnings("ignore")
    data = build()
    test = data.split(seed=0).test
    truth = data.targets(soft=True)
    model = OdorModel.load(project_root() / "models" / MODEL)
    labels = list(model.label_names)
    full = model.scorer()
    variants = {
        "ensemble": full,
        "linear only": replace(full, boosting=None, graph=None),
        "graph only": replace(full, linear=_zero_linear(full.linear), boosting=None),
        "graph, shuffled weights": replace(
            full, linear=_zero_linear(full.linear), boosting=None, graph=_shuffled(full.graph)
        ),
    }
    patterns = {note: Chem.MolFromSmarts(s) for note, s in CARRIERS.items()}

    rows = []
    for row in test:
        molecule = data.molecules[row]
        n_atoms = molecule.GetNumAtoms()
        notes = [
            note
            for note, pattern in patterns.items()
            if truth[row, labels.index(note)] >= 0.5 and molecule.HasSubstructMatch(pattern)
        ]
        if not notes:
            continue
        for variant, scorer in variants.items():
            found = explain(scorer, model.calibrator, molecule, [labels.index(n) for n in notes])
            for note, result in zip(notes, found, strict=True):
                group = {a for match in molecule.GetSubstructMatches(patterns[note]) for a in match}
                rows.append(
                    {
                        "variant": variant,
                        "note": note,
                        "smiles": data.smiles[row],
                        "hit": int(np.argmax(result.atoms)) in group,
                        "chance": len(group) / n_atoms,
                    }
                )

    table = pd.DataFrame(rows)
    out = project_root() / "results" / "attribution_check.csv"
    table.to_csv(out, index=False, float_format="%.4f")
    summary = table.groupby("variant").agg(
        cases=("hit", "size"), hit=("hit", "mean"), chance=("chance", "mean")
    )
    print(summary.round(3).to_string())
    by_note = (
        table[table.variant == "ensemble"]
        .groupby("note")
        .agg(cases=("hit", "size"), hit=("hit", "mean"), chance=("chance", "mean"))
    )
    print("\nensemble, by note\n" + by_note.round(3).to_string())
    print(f"\nwrote {out.relative_to(project_root())}")
    return 0


def _zero_linear(linear):
    """A linear part that answers the same constant for every input: its occlusions
    then contribute nothing, and the graph networks' contributions stand alone."""
    return replace(linear, weights=np.zeros_like(linear.weights))


def _shuffled(graph: GraphEnsemble) -> GraphEnsemble:
    rng = np.random.default_rng(0)
    return GraphEnsemble(
        tuple(
            GraphNetwork(
                {k: rng.permutation(v.ravel()).reshape(v.shape) for k, v in m.weights.items()},
                m.depth,
            )
            for m in graph.members
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
