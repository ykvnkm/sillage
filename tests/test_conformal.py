"""The conformal threshold keeps its promise on data it has not seen."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_conformal.py"


def load():
    spec = importlib.util.spec_from_file_location("run_conformal", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fresh_positives_clear_the_threshold_at_least_as_often_as_promised() -> None:
    conformal = load()
    rng = np.random.default_rng(0)
    scores = rng.beta(2, 5, size=(500, 1))
    positive = np.ones((500, 1), dtype=bool)

    threshold = conformal.thresholds(scores, positive, alpha=0.1)

    fresh = rng.beta(2, 5, size=20_000)
    assert 0.89 <= (fresh >= threshold[0]).mean() <= 0.92


def test_a_note_with_too_few_positives_is_always_named() -> None:
    conformal = load()
    scores = np.linspace(0.1, 0.9, 5).reshape(-1, 1)

    assert conformal.thresholds(scores, np.ones((5, 1), dtype=bool), alpha=0.1)[0] == 0.0
