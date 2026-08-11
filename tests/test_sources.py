"""Invariants every registry entry must satisfy.

These run without network access: they check the descriptions, not the data.
"""

from __future__ import annotations

import re

import pytest

from sillage.data import sources

_GITHUB_RAW = "https://raw.githubusercontent.com/"
_COMMIT_SHA = re.compile(r"\b[0-9a-f]{40}\b")
_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")

ALL_SOURCES = pytest.mark.parametrize("source", sources.REGISTRY.values(), ids=lambda s: s.name)


def test_registry_is_keyed_by_source_name() -> None:
    """A mismatch would make `get(name)` return a differently named source."""
    assert all(key == source.name for key, source in sources.REGISTRY.items())


def test_get_reports_known_names_when_asked_for_a_missing_one() -> None:
    with pytest.raises(KeyError, match="openpom-curated-gs-lf"):
        sources.get("does-not-exist")


@ALL_SOURCES
def test_url_is_https(source: sources.DataSource) -> None:
    """Plain HTTP would leave the download open to tampering in transit."""
    assert source.url.startswith("https://")


@ALL_SOURCES
def test_github_urls_address_a_commit_rather_than_a_branch(source: sources.DataSource) -> None:
    """A branch URL changes contents silently, which turns checksums into noise."""
    if source.url.startswith(_GITHUB_RAW):
        assert _COMMIT_SHA.search(source.url), f"{source.name} is pinned to a moving ref"


@ALL_SOURCES
def test_checksum_is_a_sha256_digest_when_present(source: sources.DataSource) -> None:
    if source.sha256 is not None:
        assert _SHA256.match(source.sha256), f"{source.name} has a malformed digest"


@ALL_SOURCES
def test_provenance_is_documented(source: sources.DataSource) -> None:
    """Licence and homepage are what make republishing a decision rather than a guess."""
    assert source.licence
    assert source.homepage.startswith("https://")
    assert source.description
