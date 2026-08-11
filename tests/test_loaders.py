"""End-to-end check: the real file, freshly downloaded, still honours the contract.

Marked ``network`` and therefore deselected by default. On the weekly schedule it
answers the one question no offline test can: does the pinned upstream revision
still parse and validate, or did we write a contract against data that has moved?
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sillage.data.download import fetch
from sillage.data.loaders import load_curated_openpom
from sillage.data.schemas import EXPECTED_ROW_COUNT, OPENPOM_DESCRIPTORS
from sillage.data.sources import OPENPOM_CURATED


def test_missing_file_says_how_to_get_it(tmp_path: Path) -> None:
    """The error a newcomer hits first should contain the command that fixes it."""
    with pytest.raises(FileNotFoundError, match="sillage-data fetch"):
        load_curated_openpom(tmp_path / "absent.csv")


@pytest.mark.network
def test_pinned_dataset_satisfies_the_contract(tmp_path: Path) -> None:
    result = fetch(OPENPOM_CURATED, dest_dir=tmp_path)

    frame = load_curated_openpom(result.path)

    assert len(frame) == EXPECTED_ROW_COUNT
    assert list(frame.columns[2:]) == list(OPENPOM_DESCRIPTORS)
