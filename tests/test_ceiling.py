"""Agreement arithmetic for the annotator ceiling (journal E-018).

The measurement itself needs the Pyrfume collections, which are downloaded, never
committed; these tests pin the parts that can be checked without them.
"""

from __future__ import annotations

import numpy as np
import pytest

from sillage.research.ceiling import kappas, matrix


def test_identical_raters_agree_perfectly() -> None:
    a = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])

    assert kappas(a, a) == pytest.approx([1.0, 1.0])


def test_kappa_discounts_agreement_that_chance_alone_explains() -> None:
    """Both raters say "no" to almost everything: high raw agreement, kappa near 0."""
    a = np.array([[0]] * 18 + [[1], [0]])
    b = np.array([[0]] * 18 + [[0], [1]])

    raw_agreement = (a == b).mean()
    assert raw_agreement == 0.9
    assert kappas(a, b)[0] == pytest.approx(-0.0526, abs=1e-3)


def test_kappa_is_undefined_when_a_rater_never_says_yes() -> None:
    a = np.array([[1], [0], [1]])
    b = np.array([[0], [0], [0]])

    assert np.isnan(kappas(a, b)[0])


def test_matrix_keeps_vocabulary_order() -> None:
    labels = {"A": frozenset({"woody"}), "B": frozenset({"fruity", "woody"})}

    m = matrix(labels, ["B", "A"], ("fruity", "woody", "musk"))

    assert m.tolist() == [[1, 1, 0], [0, 1, 0]]
