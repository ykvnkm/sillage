"""The README must document every source the code can download.

Documentation that is not checked stops matching the code within weeks. This is
the cheapest possible check: not that the table is beautiful, but that no source
can be added without its provenance being written down somewhere a human reads.
"""

from __future__ import annotations

import pytest

from sillage.data import sources
from sillage.paths import project_root


@pytest.fixture(scope="module")
def readme() -> str:
    return (project_root() / "README.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("source", sources.REGISTRY.values(), ids=lambda s: s.name)
def test_source_is_documented_in_the_readme(source: sources.DataSource, readme: str) -> None:
    assert source.name in readme, (
        f"{source.name} is downloadable but undocumented. "
        f"Refresh the table with: uv run sillage-data provenance"
    )
    assert source.homepage in readme
