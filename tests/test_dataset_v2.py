"""Dataset v2's label rules (journal E-019), on hand-built votes.

Building the dataset needs the Pyrfume collections, which are downloaded and never
committed; the rules that turn two experts' votes into a training target do not.
"""

from __future__ import annotations

import numpy as np
import pytest

from sillage.research import train_v2
from sillage.research.dataset_v2 import LABELS, DatasetV2, phrase_labels

NAN = np.nan


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        ("minty", {"mint"}),
        ("almond bitter almond", {"almond", "bitter"}),
        ("onion cooked onion", {"onion", "cooked"}),
        ("lily of the valley", {"lily", "muguet"}),
        ("black currant", {"black currant"}),
        ("almost odorless", {"odorless"}),
        ("bland", set()),
    ],
)
def test_goodscents_phrases_map_to_vocabulary_words(phrase: str, expected: set[str]) -> None:
    assert phrase_labels(phrase) == expected


def votes(gs: list[float], lf: list[float]) -> DatasetV2:
    """One molecule, one label per position: 1 yes, 0 no, NaN not asked."""
    width = len(LABELS)
    g = np.full((1, width), NAN)
    f = np.full((1, width), NAN)
    g[0, : len(gs)] = gs
    f[0, : len(lf)] = lf
    return DatasetV2(smiles=("C",), groups=("C",), molecules=[None], goodscents=g, leffingwell=f)


def test_targets_under_each_policy() -> None:
    # labels: both yes | both no | disagree | only GS asked (yes) | nobody asked
    d = votes(gs=[1, 0, 1, 1, NAN], lf=[1, 0, 0, NAN, NAN])

    soft = d.targets(soft=True)[0, :5]
    merge = d.targets(soft=False, mask_unasked=False)[0, :5]
    masked_hard = d.targets(soft=False)[0, :5]

    assert soft.tolist()[:4] == [1.0, 0.0, 0.5, 1.0] and np.isnan(soft[4])
    assert merge.tolist() == [1.0, 0.0, 1.0, 1.0, 0.0]  # the merge: unasked is "no"
    assert masked_hard.tolist()[:4] == [1.0, 0.0, 1.0, 1.0] and np.isnan(masked_hard[4])


def test_consensus_keeps_only_agreement_where_both_were_asked() -> None:
    d = votes(gs=[1, 0, 1, 1], lf=[1, 0, 0, NAN])

    consensus = d.consensus()[0, :4]

    assert consensus[:2].tolist() == [1.0, 0.0]
    assert np.isnan(consensus[2]) and np.isnan(consensus[3])


def test_soft_labels_pull_the_probability_towards_one_half() -> None:
    """Rows labelled 0.5 at x=1 should end near 0.5, not near 1 as a hard yes would."""
    rng = np.random.default_rng(0)
    x = np.column_stack([np.repeat([0.0, 1.0], 100), rng.normal(0, 0.01, 200)])
    soft = np.concatenate([np.zeros(100), np.full(100, 0.5)])[:, None]
    hard = np.concatenate([np.zeros(100), np.ones(100)])[:, None]
    names = ("MolWt", "TPSA")
    rows = np.arange(200)

    p_soft = train_v2.fit(x, soft, rows, names).predict_proba(np.array([[1.0, 0.0]]))[0, 0]
    p_hard = train_v2.fit(x, hard, rows, names).predict_proba(np.array([[1.0, 0.0]]))[0, 0]

    assert 0.3 < p_soft < 0.6
    assert p_hard > 0.8


def test_unasked_rows_are_left_out_of_training() -> None:
    """A label nobody asked about must not count as a negative example."""
    x = np.array([[0.0], [1.0], [1.0], [1.0]])
    y = np.array([[0.0], [1.0], [NAN], [NAN]])

    model = train_v2.fit(x, y, np.arange(4), ("MolWt",))

    assert model.constant_scores_[0] == pytest.approx(0.5)  # 1 of 2 known, not 1 of 4


def test_score_skips_descriptors_without_enough_known_examples() -> None:
    truth = np.array([[1, NAN]] * 6 + [[0, NAN]] * 6, dtype=float)
    predicted = np.column_stack([np.linspace(1, 0, 12), np.zeros(12)])

    result = train_v2.score(predicted, truth)

    assert result["descriptors"] == 1
    assert result["macro_ap"] == pytest.approx(1.0)
