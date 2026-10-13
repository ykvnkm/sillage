"""Measure the annotator-agreement ceiling and where the model sits under it.

    uv run sillage-data fetch                      # Pyrfume collections, pinned
    uv run python scripts/run_ceiling.py           # results/annotator_ceiling_<model>.{csv,json}
    SILLAGE_MODEL_DIR=models/odor-baseline-v1 uv run python scripts/run_ceiling.py

See sillage.research.ceiling for the method and journal E-018 for the reading.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

from sillage.features import parse_smiles
from sillage.features import sets as feature_sets
from sillage.models.artifact import OdorModel
from sillage.models.compiled import LinearMultilabel
from sillage.paths import model_dir, project_root
from sillage.research.ceiling import MIN_POSITIVES, kappas, load, matrix


def main() -> int:
    collections = load()
    vocabulary = collections.vocabulary
    shared = collections.shared_molecules
    print(f"shared vocabulary: {len(vocabulary)} descriptors; molecules in both: {len(shared)}")

    gs_all = matrix(collections.goodscents, shared, vocabulary)
    lf_all = matrix(collections.leffingwell, shared, vocabulary)
    human_all = kappas(gs_all, lf_all)

    # Molecules the model never saw: the test split of its own artifact.
    model = OdorModel.load(model_dir())
    test = {r.smiles for r in model.reference if r.split == "test"}
    unseen = [m for m in shared if m in test]
    gs = matrix(collections.goodscents, unseen, vocabulary)
    lf = matrix(collections.leffingwell, unseen, vocabulary)

    features = feature_sets.build(model.card.feature_set, parse_smiles(unseen))
    raw = LinearMultilabel.from_classifier(model.estimator).predict_proba(features.values)
    probability = model.calibrator.transform(raw)
    columns = [model.label_names.index(d) for d in vocabulary]
    said = (probability[:, columns] >= 0.5).astype(np.int8)

    table = pd.DataFrame(
        {
            "descriptor": vocabulary,
            "experts_kappa_all": human_all,
            "experts_kappa_unseen": kappas(gs, lf),
            "model_vs_leffingwell": kappas(said, lf),
            "model_vs_goodscents": kappas(said, gs),
            "positives_gs_unseen": gs.sum(axis=0),
            "positives_lf_unseen": lf.sum(axis=0),
            "positives_gs_all": gs_all.sum(axis=0),
            "positives_lf_all": lf_all.sum(axis=0),
        }
    )
    out = project_root() / "results" / f"annotator_ceiling_{model.card.name}.csv"
    table.to_csv(out, index=False, float_format="%.4f")

    usable = (table.positives_gs_unseen >= MIN_POSITIVES) & (
        table.positives_lf_unseen >= MIN_POSITIVES
    )
    stable = (table.positives_gs_all >= MIN_POSITIVES) & (table.positives_lf_all >= MIN_POSITIVES)
    print(f"unseen molecules in both collections: {len(unseen)}")
    print(f"descriptors with >= {MIN_POSITIVES} positives in each, unseen: {int(usable.sum())}")
    print()
    print("macro kappa (descriptors with enough positives)")
    experts_all = table.experts_kappa_all[stable].mean()
    print(f"  experts, all {len(shared)} shared molecules  {experts_all:.3f}")
    u = table[usable]
    print(f"  experts, unseen molecules       {u.experts_kappa_unseen.mean():.3f}")
    print(f"  model vs Leffingwell, unseen    {u.model_vs_leffingwell.mean():.3f}")
    print(f"  model vs Goodscents, unseen     {u.model_vs_goodscents.mean():.3f}")

    # Per descriptor: does the model do better where the experts agree? Model quality
    # as AP on every unseen molecule (not only the 233 in both collections), against
    # experts' kappa on all 2352 shared molecules: both sides measured on enough data.
    all_test = [r for r in model.reference if r.split == "test"]
    x_all = feature_sets.build(model.card.feature_set, parse_smiles([r.smiles for r in all_test]))
    p_all = LinearMultilabel.from_classifier(model.estimator).predict_proba(x_all.values)
    y_all = np.array([[d in r.descriptors for d in model.label_names] for r in all_test], dtype=int)
    pairs = []
    for _, row in table[stable].iterrows():
        j = model.label_names.index(row.descriptor)
        if y_all[:, j].sum() >= 3:
            pairs.append((row.experts_kappa_all, average_precision_score(y_all[:, j], p_all[:, j])))
    experts_k, model_ap = zip(*pairs, strict=True)
    rho_ap, p_ap = spearmanr(experts_k, model_ap)
    print(
        f"\nper descriptor ({len(pairs)}): Spearman of experts' kappa vs model AP = "
        f"{rho_ap:.2f} (p={p_ap:.1g})"
    )

    ordered = table[stable].sort_values("experts_kappa_all")
    print("\nexperts agree least on:")
    print(
        ordered.head(8)[["descriptor", "experts_kappa_all", "model_vs_leffingwell"]].to_string(
            index=False
        )
    )
    print("\nexperts agree most on:")
    print(
        ordered.tail(8)[["descriptor", "experts_kappa_all", "model_vs_leffingwell"]].to_string(
            index=False
        )
    )
    summary = {
        "model": model.card.name,
        "descriptors": int(usable.sum()),
        "molecules": len(unseen),
        "experts": round(float(u.experts_kappa_unseen.mean()), 3),
        "model_vs_leffingwell": round(float(u.model_vs_leffingwell.mean()), 3),
        "model_vs_goodscents": round(float(u.model_vs_goodscents.mean()), 3),
        "spearman_experts_vs_model_ap": round(float(rho_ap), 2),
    }
    summary_path = project_root() / "results" / f"annotator_ceiling_{model.card.name}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", "utf-8")
    print(f"\nwrote {out.relative_to(project_root())}, {summary_path.relative_to(project_root())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
