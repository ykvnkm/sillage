"""An odour profile, and the words the product is allowed to use about it.

The confidence bands moved here from ``sillage.models.calibration`` in step 2.3.
They are not a property of logistic regression or of Platt scaling: they are the
product's promise about what "likely" means, and the GNN will be held to the same
promise. A promise belongs to the domain.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum


class Confidence(StrEnum):
    """Qualitative verdict for a single descriptor.

    The product shows these words rather than numbers, and the band edges come from
    measurement rather than taste. Observed frequency on the unseen test split, over
    calibrated labels only (journal E-011):

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

UNVERIFIED_MENTION = 0.5
"""Score at which an uncalibrated descriptor is mentioned at all.

It is the classifier's own decision boundary and nothing more: crossing it means
"the model leans towards yes", not "yes with probability 0.5". On the test split
these labels promise 0.77 and deliver 0.36 (journal E-011), which is exactly why
they are reported as a separate, numberless list rather than ranked among the
calibrated ones."""

_STRENGTH = {verdict: len(BANDS) - index for index, (_, verdict) in enumerate(BANDS)}


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


@dataclass(frozen=True, slots=True)
class DescriptorScore:
    """What the model says about one descriptor for one molecule."""

    name: str
    score: float
    """Calibrated probability when :attr:`calibrated`, a bare ranking score otherwise.
    Read :attr:`probability` instead unless you need the ordering."""

    calibrated: bool
    base_rate: float
    """How often the descriptor occurs in training data. "Musk, possible" means more
    next to "musk occurs in 1% of molecules" than on its own."""

    @property
    def confidence(self) -> Confidence:
        return confidence_of(self.score, calibrated=self.calibrated)

    @property
    def probability(self) -> float | None:
        """The number that may be shown to a user, or None when there is none."""
        return self.score if self.calibrated else None

    def at_least(self, threshold: Confidence) -> bool:
        """True for a calibrated verdict at or above *threshold*."""
        return _STRENGTH.get(self.confidence, 0) >= _STRENGTH[threshold]


@dataclass(frozen=True, slots=True)
class OdorProfile:
    """Scores for every descriptor the model knows, in the model's label order."""

    scores: tuple[DescriptorScore, ...]

    @classmethod
    def from_scores(cls, scores: Iterable[DescriptorScore]) -> OdorProfile:
        return cls(scores=tuple(scores))

    def confident(self, at_least: Confidence = Confidence.POSSIBLE) -> tuple[DescriptorScore, ...]:
        """Calibrated descriptors at or above a band, most probable first.

        Only these may be ranked against each other by number.
        """
        kept = [s for s in self.scores if s.calibrated and s.at_least(at_least)]
        return tuple(sorted(kept, key=lambda s: s.score, reverse=True))

    def unverified(self) -> tuple[DescriptorScore, ...]:
        """Uncalibrated descriptors the model leans towards, in its own ranking order.

        Kept apart from :meth:`confident` on purpose. Sorted together, an uncalibrated
        0.59 would outrank a calibrated 0.58 while being right half as often.
        """
        kept = [s for s in self.scores if not s.calibrated and s.score >= UNVERIFIED_MENTION]
        return tuple(sorted(kept, key=lambda s: s.score, reverse=True))

    def __getitem__(self, name: str) -> DescriptorScore:
        for score in self.scores:
            if score.name == name:
                return score
        raise KeyError(name)
