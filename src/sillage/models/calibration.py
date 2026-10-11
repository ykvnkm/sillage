"""Turning raw scores into probabilities that mean what they say.

Measured on the validation split, the uncalibrated model is over-confident exactly
where it matters: when it outputs 0.94 the descriptor applies 83% of the time. The
gap is invisible in a pooled average because 92% of pairs sit below 0.1.

The method is **Platt scaling, fitted per label**: a two-parameter sigmoid over the
model's log-odds, learnt on held-out data. Two parameters and not more, on purpose.
The rarest descriptor has three positive examples in validation; isotonic regression
would fit that noise exactly and call it calibration.

Per label and not globally, because the labels are not alike. `fruity` has 189
positives in validation and `chamomile` three; one shared correction would be
dominated by the frequent descriptors, and the rare ones -- the ones a user is most
likely to be misled about -- would stay as wrong as before.

A deliberate consequence of mild regularisation: where validation evidence is weak,
the slope shrinks toward zero and the output moves toward the label's base rate.
That is the honest answer for a descriptor the model has barely seen, and it is why
the regularisation is not turned off.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from sklearn.linear_model import LogisticRegression

CLIP = 1e-6
"""Raw scores reach exactly 0 and 1, whose log-odds are infinite."""

MIN_POSITIVES = 6
"""Measured, not guessed. With 3-5 positive examples in validation, Platt scaling
fits the noise: on the test split those labels promise 0.772 and deliver 0.364, a gap
of +0.409. From 6 positives upward the gap falls to 0.04 or less (journal E-011).

31 of 138 descriptors fall below this line."""


@dataclass(frozen=True, slots=True)
class PerLabelCalibrator:
    """Platt scaling per label, plus a base-rate fallback where it could not be fitted."""

    slopes: np.ndarray
    intercepts: np.ndarray
    base_rates: np.ndarray
    """Training prevalence per label. Not used by :meth:`transform`; carried so the
    product can show a descriptor's background frequency next to its score."""

    fitted: np.ndarray
    """Boolean per label. ``False`` means :meth:`transform` passes the raw score
    through unchanged, and the caller **must not present it as a probability**.

    Passing the raw score through rather than substituting the base rate is
    deliberate. A constant output would destroy the ranking inside those 31 labels
    and drop macro AP, discarding information the model genuinely has. The ordering
    is worth keeping; only the claim "this is a probability" is not."""

    @property
    def n_labels(self) -> int:
        return int(self.slopes.size)

    @property
    def n_fitted(self) -> int:
        return int(self.fitted.sum())

    def transform(self, raw_scores: np.ndarray) -> np.ndarray:
        """Map raw scores to calibrated probabilities, column by column."""
        scores = np.asarray(raw_scores, dtype=np.float64)
        if scores.ndim != 2:
            raise ValueError(f"Expected a 2-D score matrix, got {scores.ndim}-D")
        if scores.shape[1] != self.n_labels:
            raise ValueError(f"Calibrator has {self.n_labels} labels, got {scores.shape[1]}")

        logits = _log_odds(scores)
        calibrated = 1.0 / (1.0 + np.exp(-(logits * self.slopes + self.intercepts)))
        # Identity for uncalibrated labels: both branches are monotone per column, so
        # per-label AP and AUC are preserved exactly. A test asserts that.
        return np.where(self.fitted, calibrated, scores)

    @classmethod
    def fit(
        cls,
        raw_scores: np.ndarray,
        labels: np.ndarray,
        *,
        base_rates: np.ndarray,
        c: float = 1.0,
    ) -> PerLabelCalibrator:
        """Fit on held-out scores -- never on data the model was trained on.

        Calibrating on training data produces a model that looks perfectly calibrated
        and is not: it has already memorised those answers, so its scores there are
        far better than on anything new.
        """
        scores = np.asarray(raw_scores, dtype=np.float64)
        truth = np.asarray(labels)
        if scores.shape != truth.shape:
            raise ValueError(f"Scores have shape {scores.shape}, labels {truth.shape}")

        n_labels = scores.shape[1]
        if base_rates.shape != (n_labels,):
            raise ValueError(f"Expected {n_labels} base rates, got {base_rates.shape}")

        logits = _log_odds(scores)
        slopes = np.zeros(n_labels)
        intercepts = np.zeros(n_labels)
        fitted = np.zeros(n_labels, dtype=bool)

        for index in range(n_labels):
            column = truth[:, index]
            positives = int(column.sum())
            if positives < MIN_POSITIVES or positives == column.size:
                continue  # left uncalibrated; transform passes the raw score through

            model = LogisticRegression(C=c).fit(logits[:, index].reshape(-1, 1), column)
            slope = float(model.coef_[0, 0])
            # A negative slope means validation says this label's scores are inverted.
            # With a handful of positives that is noise, not signal, and accepting it
            # would silently reverse the ranking the ablation measured.
            if slope <= 0.0:
                continue

            slopes[index] = slope
            intercepts[index] = float(model.intercept_[0])
            fitted[index] = True

        return cls(
            slopes=slopes,
            intercepts=intercepts,
            base_rates=np.asarray(base_rates, dtype=np.float64),
            fitted=fitted,
        )


class Confidence(StrEnum):
    """Qualitative verdict for a single descriptor.

    The product shows these words rather than numbers, and the band edges come from
    measurement rather than taste. Observed frequency on the unseen test split, over
    calibrated labels only:

    ====================  =========  ==========
    Band                  Promised   Observed
    ====================  =========  ==========
    ``LIKELY``   >= 0.85      0.903       0.889
    ``PROBABLE`` 0.55-0.85    0.640       0.667
    ``POSSIBLE`` 0.35-0.55    0.443       0.446
    ``UNLIKELY`` 0.15-0.35    0.233       0.217
    ====================  =========  ==========

    There is no band above ``LIKELY``. Separating 0.90 from 0.97 would need far more
    than the 99 pairs available up there, so claiming the difference would be
    claiming something unmeasured.
    """

    LIKELY = "likely"
    PROBABLE = "probable"
    POSSIBLE = "possible"
    UNLIKELY = "unlikely"
    NEGLIGIBLE = "negligible"

    UNVERIFIED = "unverified"
    """The descriptor has too few examples to calibrate (31 of 138). The score still
    ranks sensibly -- macro AP is unchanged for these labels -- but it is not a
    probability, and no number should be shown for it."""


BANDS: tuple[tuple[float, Confidence], ...] = (
    (0.85, Confidence.LIKELY),
    (0.55, Confidence.PROBABLE),
    (0.35, Confidence.POSSIBLE),
    (0.15, Confidence.UNLIKELY),
)


def confidence_of(probability: float, *, calibrated: bool) -> Confidence:
    """Translate a probability into a verdict the interface can show.

    An uncalibrated label returns :attr:`Confidence.UNVERIFIED` whatever its score:
    the number is a ranking position, not a claim about frequency.
    """
    if not calibrated:
        return Confidence.UNVERIFIED
    for edge, verdict in BANDS:
        if probability >= edge:
            return verdict
    return Confidence.NEGLIGIBLE


def _log_odds(scores: np.ndarray) -> np.ndarray:
    clipped = np.clip(scores, CLIP, 1.0 - CLIP)
    return np.log(clipped / (1.0 - clipped))
