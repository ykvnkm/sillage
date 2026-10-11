"""Measuring whether a probability means what it says.

A score of 0.9 should be right nine times out of ten. Average precision cannot tell
you whether it is: it only cares about order, so dividing every score by a thousand
leaves it untouched. Calibration is the separate question, and it needs its own
measurement.

**Equal-width bins lie on this data, and for the same reason micro AUC did.** The
label matrix is 3.5% ones, so 92% of all (molecule, descriptor) pairs score below
0.1, where the model is accurate. Averaging over bins of equal width lets that mass
drown everything else: pooled ECE comes out at 0.0034 while the top bin promises
0.94 and delivers 0.83.

Equal-mass bins give every bin the same number of pairs, so the confident end is
visible. Both are reported, because the gap between them is itself the finding.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean squared error of the probabilities. Lower is better.

    Sensitive to both ranking and calibration at once, which makes it a decent
    single summary and a poor diagnostic -- it cannot say which of the two moved.
    """
    truth, probability = _validated(y_true, y_prob)
    return float(np.mean((probability - truth) ** 2))


def reliability_table(
    y_true: np.ndarray, y_prob: np.ndarray, *, bins: int = 10, equal_mass: bool = True
) -> pd.DataFrame:
    """What the model promised against what happened, bin by bin.

    Args:
        equal_mass: bins holding the same number of pairs (the default), rather than
            the same width. On sparse labels equal width hides the confident end.

    Returns:
        One row per non-empty bin: pair count, mean predicted probability, observed
        frequency, and the signed gap. Positive gap means over-confidence.
    """
    truth, probability = _validated(y_true, y_prob)
    edges = _bin_edges(probability, bins=bins, equal_mass=equal_mass)

    rows = []
    for low, high in pairwise(edges):
        inside = (probability >= low) & (probability < high)
        if high >= edges[-1]:
            inside = (probability >= low) & (probability <= high)
        if not inside.any():
            continue
        predicted = float(probability[inside].mean())
        observed = float(truth[inside].mean())
        rows.append(
            {
                "low": float(low),
                "high": float(high),
                "pairs": int(inside.sum()),
                "predicted": predicted,
                "observed": observed,
                "gap": predicted - observed,
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, *, bins: int = 10, equal_mass: bool = True
) -> float:
    """Average absolute gap between promise and reality, weighted by bin size."""
    table = reliability_table(y_true, y_prob, bins=bins, equal_mass=equal_mass)
    if table.empty:
        return float("nan")
    weights = table["pairs"].to_numpy() / table["pairs"].sum()
    return float(np.sum(weights * table["gap"].abs().to_numpy()))


def calibration_at_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> tuple[int, float, float]:
    """Among predictions at or above *threshold*, what was promised and what happened.

    The measurement this project actually relies on, because binning fails here in
    both of its usual forms. Equal-width bins let the 92% of near-zero pairs drown
    the confident end. Equal-mass bins collapse onto each other, since those pairs
    share almost identical values and quantile edges land on top of one another.

    A threshold needs no bins and answers the question a user has: when the service
    says "likely", how often is it right?

    Returns:
        Pair count, mean promised probability, observed frequency. A count of zero
        returns NaN for both means rather than inventing a number.
    """
    truth, probability = _validated(y_true, y_prob)
    selected = probability >= threshold
    if not selected.any():
        return 0, float("nan"), float("nan")
    return (
        int(selected.sum()),
        float(probability[selected].mean()),
        float(truth[selected].mean()),
    )


def _bin_edges(probability: np.ndarray, *, bins: int, equal_mass: bool) -> np.ndarray:
    if not equal_mass:
        return np.linspace(0.0, 1.0, bins + 1)
    quantiles = np.quantile(probability, np.linspace(0.0, 1.0, bins + 1))
    # Ties collapse quantiles onto each other; duplicated edges would make empty bins.
    return np.unique(quantiles)


def _validated(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(y_true).ravel().astype(np.float64)
    probability = np.asarray(y_prob, dtype=np.float64).ravel()
    if truth.size != probability.size:
        raise ValueError(f"{truth.size} labels against {probability.size} probabilities")
    if truth.size == 0:
        raise ValueError("Cannot measure calibration on an empty set")
    if not np.isin(truth, (0.0, 1.0)).all():
        raise ValueError("Labels must be binary")
    return truth, probability
