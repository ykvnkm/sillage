"""The two baseline models of phase 1, and the trivial one they must beat.

Both are wrapped in a pipeline rather than used bare, because the preprocessing has
to be *fitted*, and fitting means it may only ever see the training split. Putting
the scaler inside the pipeline is what makes that automatic instead of a thing to
remember (see ADR-0006 for the same boundary drawn on the featurisation side).

Scaling is not a formality for the linear model. Measured on this dataset: without
it, logistic regression fails to converge in 2000 iterations and takes 2.8s per
label; with it, it converges in 0.11s. Binary fingerprint bits and a molecular
weight of 1347 cannot share an optimisation landscape.
"""

from __future__ import annotations

from collections.abc import Sequence

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, StandardScaler

DEFAULT_SEED = 0
FINGERPRINT_PREFIX = "morgan_"


def _classifier(c: float, seed: int) -> LogisticRegression:
    """``C`` is the inverse of regularisation strength: smaller means simpler.

    0.1 is deliberately conservative. With 2264 features against 3967 molecules and
    31 positives for the rarest label, the model can memorise its way to a perfect
    training score without learning anything.
    """
    return LogisticRegression(C=c, max_iter=1000, random_state=seed, solver="lbfgs")


def logistic_regression(
    feature_names: Sequence[str], *, c: float = 0.1, seed: int = DEFAULT_SEED
) -> Pipeline:
    """Logistic regression with each feature block scaled the way it needs.

    Fingerprint columns and descriptor columns want different treatment, and using
    one scaler for both measures the scaler rather than the features (journal E-005).

    *Fingerprints* get ``MaxAbsScaler``: they are already on a common 0/1 scale, and
    dividing by the standard deviation would blow up the rare bits. A bit set in 5
    of 3967 molecules has a standard deviation near 0.035, so standardising sends
    those five rows to about 28 while ordinary features live within +/-3.

    *Descriptors* get ``StandardScaler``: molecular weight in the hundreds next to a
    ring count in the single digits is exactly what standardisation is for, and
    without it the optimiser does not converge at all.

    Blocks are found by column name, which is why :class:`FeatureMatrix` carries its
    names alongside its values.
    """
    fingerprints = [
        i for i, name in enumerate(feature_names) if name.startswith(FINGERPRINT_PREFIX)
    ]
    others = [i for i, name in enumerate(feature_names) if not name.startswith(FINGERPRINT_PREFIX)]

    blocks = []
    if fingerprints:
        blocks.append(("fingerprints", MaxAbsScaler(), fingerprints))
    if others:
        blocks.append(("descriptors", StandardScaler(), others))

    return Pipeline([("scale", ColumnTransformer(blocks)), ("model", _classifier(c, seed))])


def logistic_regression_single_scaler(
    feature_names: Sequence[str],  # unused: signature shared with the block-wise variant
    *,
    scaler: str = "standard",
    c: float = 0.1,
    seed: int = DEFAULT_SEED,
) -> Pipeline:
    """One scaler for the whole matrix. Kept to reproduce the finding in E-005."""
    chosen = StandardScaler() if scaler == "standard" else MaxAbsScaler()
    return Pipeline([("scale", chosen), ("model", _classifier(c, seed))])


def gradient_boosting(
    *, max_iter: int = 150, learning_rate: float = 0.1, seed: int = DEFAULT_SEED
) -> HistGradientBoostingClassifier:
    """Histogram-based gradient boosting.

    No scaler: trees split on thresholds, so monotone rescaling of a feature changes
    nothing. Included because it differs from the linear model in the way that
    matters most here -- it can represent interactions between fragments, which a
    linear model cannot. If the two land on the same number, the ceiling is set by
    the features rather than by the model.
    """
    return HistGradientBoostingClassifier(
        max_iter=max_iter,
        learning_rate=learning_rate,
        random_state=seed,
        early_stopping=False,
    )
