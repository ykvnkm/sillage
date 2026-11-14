"""Does the ensemble keep its promises? Reliability by band, with error bars.

    uv run python scripts/run_reliability.py     # ~1 min; needs the cached training jobs

The model card checks the top band on the test part: 104 pairs, promised 0.90,
observed 0.85. A hundred pairs cannot tell a real gap from luck, so this measures the
same thing on two sets and puts an interval on each number:

  test    the shipped model and its calibrator on the 570 test molecules;
  cross   the 5021 learning molecules, each scored by the fold models that never saw
          it and calibrated by a calibrator fitted on the other four folds.

Intervals are bootstrap over molecules, not over pairs: one molecule contributes
several notes, and they stand or fall together. Journal E-023.
"""

from __future__ import annotations

import warnings

import joblib
import numpy as np
import pandas as pd

from sillage.models import PerLabelCalibrator
from sillage.models.artifact import OdorModel
from sillage.paths import model_dir, project_root
from sillage.research.dataset_v2 import build
from sillage.research.train_model_v2 import FOLDS
from sillage.research.train_model_v3 import _folds

BANDS = {
    "likely": (0.85, 1.01),
    "probable": (0.55, 0.85),
    "possible": (0.35, 0.55),
    "unlikely": (0.15, 0.35),
    "below": (0.0, 0.15),
}
SEED = 0
RESAMPLES = 2000


def main() -> int:
    data = build()
    split = data.split(seed=SEED)
    targets = data.targets(soft=True)
    learn = np.concatenate([split.train, split.val])
    folds = _folds(learn, data.groups, SEED)
    interim = project_root() / "data" / "interim"

    raw = np.zeros((learn.size, targets.shape[1]))
    for k in range(FOLDS):
        raw[folds == k] = joblib.load(interim / f"ensemble_job_{k}_seed{SEED}.joblib")[1]
    truth = targets[learn]
    with warnings.catch_warnings():  # a label nobody was asked about has no mean
        warnings.simplefilter("ignore", RuntimeWarning)
        base_rates = np.nan_to_num(np.nanmean(truth, axis=0), nan=0.0)
    cross = np.full_like(raw, np.nan)
    for k in range(FOLDS):
        other = folds != k
        calibrator = PerLabelCalibrator.fit(raw[other], truth[other], base_rates=base_rates)
        cross[folds == k] = np.where(
            calibrator.fitted, calibrator.transform(raw[folds == k]), np.nan
        )

    model = OdorModel.load(model_dir())
    test_raw = model.scorer().raw([data.molecules[i] for i in split.test])
    test = np.where(model.calibrator.fitted, model.calibrator.transform(test_raw), np.nan)

    table = pd.concat(
        [
            _bands("test", test, targets[split.test]),
            _bands("cross", cross, truth),
        ]
    )
    out = project_root() / "results" / f"reliability_{model.card.name}.csv"
    table.to_csv(out, index=False, float_format="%.4f")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nwrote {out.relative_to(project_root())}")
    figure = project_root() / "docs" / "images" / f"reliability-{model.card.name}.png"
    _draw(table, figure)
    print(f"wrote {figure.relative_to(project_root())}")
    return 0


RUSSIAN = {
    "likely": "очень вероятно\n0.85–1.00",
    "probable": "вероятно\n0.55–0.85",
    "possible": "возможно\n0.35–0.55",
    "unlikely": "маловероятно\n0.15–0.35",
}
SERIES = {
    "cross": ("перекрёстная проверка, 5021 молекула", "#2a78d6"),
    "test": ("тест, 570 молекул", "#eb6834"),
}


def _draw(table: pd.DataFrame, path) -> None:
    """The gap per band with its interval: a dot on zero is a kept promise."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, surface = "#0b0b0b", "#52514e", "#fcfcfb"
    figure, axes = plt.subplots(figsize=(8, 4.2), dpi=200, facecolor=surface)
    axes.set_facecolor(surface)
    axes.axhline(0, color=muted, linewidth=1)
    bands = list(RUSSIAN)
    for shift, (name, (label, colour)) in zip((-0.12, 0.12), SERIES.items(), strict=True):
        rows = table[table["set"] == name].set_index("band").loc[bands]
        x = np.arange(len(bands)) + shift
        gap = rows["gap"].to_numpy() * 100
        low, high = rows["gap_low"].to_numpy() * 100, rows["gap_high"].to_numpy() * 100
        axes.vlines(x, low, high, color=colour, linewidth=2)
        axes.plot(x, gap, "o", color=colour, markersize=8, markeredgecolor=surface, label=label)
        for xi, pairs, top in zip(x, rows["pairs"], high, strict=True):
            axes.annotate(
                f"{pairs} пар", (xi, top), xytext=(0, 5), textcoords="offset points",
                ha="center", fontsize=8, color=muted,
            )  # fmt: skip
    axes.set_xticks(np.arange(len(bands)), [RUSSIAN[b] for b in bands], color=ink)
    axes.set_ylabel("обещано − сбылось, п.п.", color=muted)
    axes.set_title(
        "Где пар больше пятисот, обещание расходится с результатом на 2 п.п. и меньше",
        loc="left", fontsize=11, color=ink,
    )  # fmt: skip
    axes.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    axes.set_axisbelow(True)
    axes.tick_params(colors=muted, length=0)
    for side in axes.spines.values():
        side.set_visible(False)
    axes.legend(frameon=False, loc="upper right", labelcolor=ink, fontsize=9)
    figure.tight_layout()
    figure.savefig(path, facecolor=surface)
    plt.close(figure)


def _bands(name: str, probability: np.ndarray, truth: np.ndarray) -> pd.DataFrame:
    """Promised against observed per band; NaN probability or truth is "not measured"."""
    rng = np.random.default_rng(SEED)
    usable = ~np.isnan(probability) & ~np.isnan(truth)
    rows = []
    for band, (low, high) in BANDS.items():
        inside = usable & (probability >= low) & (probability < high)
        # Per molecule: pairs in the band, promised sum, observed sum.
        pairs = inside.sum(axis=1)
        promised = np.where(inside, probability, 0.0).sum(axis=1)
        observed = np.where(inside, truth, 0.0).sum(axis=1)
        draws = rng.integers(0, len(pairs), size=(RESAMPLES, len(pairs)))
        gaps = (promised[draws].sum(axis=1) - observed[draws].sum(axis=1)) / np.maximum(
            pairs[draws].sum(axis=1), 1
        )
        rows.append(
            {
                "set": name,
                "band": band,
                "pairs": int(pairs.sum()),
                "promised": promised.sum() / pairs.sum(),
                "observed": observed.sum() / pairs.sum(),
                "gap": (promised.sum() - observed.sum()) / pairs.sum(),
                "gap_low": float(np.quantile(gaps, 0.025)),
                "gap_high": float(np.quantile(gaps, 0.975)),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    raise SystemExit(main())
