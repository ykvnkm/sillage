"""Does a graph network beat the fingerprints, and what does it add to them? (journal E-020)

    uv sync --group gnn
    uv run python scripts/run_gnn.py          # ~1.5 hours on a laptop CPU; results/gnn.csv

Dataset v2, soft masked labels, the same five skeleton-grouped splits as E-019. Every
model trains on the training part; the graph networks stop on the validation part;
all are scored on the same unseen molecules against experts' consensus and each
collection. Variants:

  LR              the shipping baseline (Morgan r3 + descriptors, C=0.1)
  GB              boosting on the descriptors
  GNN             one D-MPNN
  GNN x5          five D-MPNNs from different seeds, averaged
  GNN x5 flat     the same, with every stereo label blanked -- what stereo is worth
  LR + GB         and every mix of the above with the graph networks: plain means

Test predictions are saved to data/interim/ (derived from derived-only records, so
never committed) for any later look that should not need another hour of training.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

from sillage.features import sets as feature_sets
from sillage.features.graphs import ATOM_STEREO_COLUMNS, BOND_STEREO_COLUMNS, molecular_graph
from sillage.paths import project_root
from sillage.research import gnn, train_v2
from sillage.research.dataset_v2 import build

SEEDS = (0, 1, 2, 3, 4)
MEMBERS = 5
# Chosen on validation only, splits 0 and 1 (journal E-020): wider, faster, longer than
# Chemprop's defaults, which stopped improving only at the last epoch on this data.
CONFIG = gnn.TrainingConfig(hidden=600, learning_rate=2e-3, epochs=100, patience=25)


def main() -> int:
    warnings.filterwarnings("ignore")
    data = build()
    graphs = [molecular_graph(m) for m in data.molecules]
    flat = [_without_stereo(g) for g in graphs]
    full = feature_sets.build("morgan-bin-r3+desc", data.molecules)
    desc = feature_sets.build("descriptors", data.molecules)
    targets = data.targets(soft=True)
    experts = data.targets(soft=False)
    truths = {
        "consensus": data.consensus(),
        "leffingwell": data.leffingwell,
        "goodscents": data.goodscents,
    }
    interim = project_root() / "data" / "interim"
    interim.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in SEEDS:
        started = time.time()
        split = data.split(seed=seed)
        test, val = split.test, split.val
        lr = train_v2.fit(full.values, targets, split.train, full.names, seed=seed)
        gb = train_v2.fit_boosting(desc, targets, split.train, seed=seed)
        p = {
            "LR": lr.predict_proba(full.values[test]),
            "GB": gb.predict_proba(desc.values[test]),
        }
        for name, these in [("GNN x5", graphs), ("GNN x5 flat", flat)]:
            members = []
            for member in range(MEMBERS):
                trained = gnn.train(
                    these, targets, split.train, val, experts[val],
                    config=CONFIG, seed=seed * 100 + member,
                )  # fmt: skip
                members.append(gnn.predict(trained.model, these, test))
            p[name] = np.mean(members, axis=0)
            if name == "GNN x5":
                p["GNN"] = members[0]

        p["LR + GB"] = (p["LR"] + p["GB"]) / 2
        p["GNN x5 + LR"] = (p["GNN x5"] + p["LR"]) / 2
        p["GNN x5 + LR + GB"] = (p["GNN x5"] + p["LR"] + p["GB"]) / 3
        p["GNN x5 flat + LR + GB"] = (p["GNN x5 flat"] + p["LR"] + p["GB"]) / 3

        np.savez_compressed(interim / f"gnn_test_predictions_seed{seed}.npz", test=test, **p)
        for name, predicted in p.items():
            for truth_name, truth in truths.items():
                result = train_v2.score(predicted, truth[test])
                rows.append({"variant": name, "seed": seed, "truth": truth_name, **result})
        print(f"seed {seed} done in {time.time() - started:.0f}s", flush=True)

    table = pd.DataFrame(rows)
    out = project_root() / "results" / "gnn.csv"
    table.to_csv(out, index=False, float_format="%.4f")
    print(table.groupby(["truth", "variant"]).macro_ap.agg(["mean", "std"]).round(4).to_string())
    print(f"\nwrote {out.relative_to(project_root())}")
    return 0


def _without_stereo(graph):
    atoms = graph.atoms.copy()
    bonds = graph.bond_features.copy()
    atoms[:, ATOM_STEREO_COLUMNS] = 0.0
    bonds[:, BOND_STEREO_COLUMNS] = 0.0
    return type(graph)(atoms=atoms, bond_features=bonds, bonds=graph.bonds)


if __name__ == "__main__":
    raise SystemExit(main())
