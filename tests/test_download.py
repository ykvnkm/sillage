"""Download behaviour, exercised offline.

The fetcher is pointed at ``file://`` URLs, so the full code path -- streaming,
digesting, atomic rename -- runs without touching the network. Tests that do reach
the internet are marked ``network`` and deselected by default.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from sillage.data import sources
from sillage.data.download import ChecksumMismatchError, fetch, sha256_of
from sillage.data.sources import DataSource, Redistribution

PAYLOAD = b"smiles,descriptor\nCCO,alcoholic\n"
PAYLOAD_SHA256 = hashlib.sha256(PAYLOAD).hexdigest()


@pytest.fixture
def remote_file(tmp_path: Path) -> Path:
    """Stands in for an external server, addressable over file://."""
    path = tmp_path / "remote" / "dataset.csv"
    path.parent.mkdir()
    path.write_bytes(PAYLOAD)
    return path


def make_source(remote: Path, *, sha256: str | None) -> DataSource:
    return DataSource(
        name="fixture",
        url=remote.as_uri(),
        filename="dataset.csv",
        sha256=sha256,
        licence="MIT",
        homepage="https://example.invalid",
        redistribution=Redistribution.ALLOWED,
        description="fixture",
    )


def test_sha256_of_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(PAYLOAD)
    assert sha256_of(path) == PAYLOAD_SHA256


def test_fetch_downloads_and_verifies(remote_file: Path, tmp_path: Path) -> None:
    result = fetch(make_source(remote_file, sha256=PAYLOAD_SHA256), dest_dir=tmp_path / "raw")

    assert result.downloaded is True
    assert result.sha256 == PAYLOAD_SHA256
    assert result.path.read_bytes() == PAYLOAD


def test_fetch_reuses_a_valid_local_copy_without_touching_the_source(
    remote_file: Path, tmp_path: Path
) -> None:
    """Idempotence: deleting the remote afterwards proves the second run stayed local."""
    source = make_source(remote_file, sha256=PAYLOAD_SHA256)
    dest = tmp_path / "raw"
    fetch(source, dest_dir=dest)
    remote_file.unlink()

    result = fetch(source, dest_dir=dest)

    assert result.downloaded is False
    assert result.path.read_bytes() == PAYLOAD


def test_fetch_rejects_a_source_that_changed_under_us(remote_file: Path, tmp_path: Path) -> None:
    wrong_digest = "0" * 64
    source = make_source(remote_file, sha256=wrong_digest)

    with pytest.raises(ChecksumMismatchError) as error:
        fetch(source, dest_dir=tmp_path / "raw")

    assert error.value.actual == PAYLOAD_SHA256


def test_a_rejected_download_leaves_nothing_behind(remote_file: Path, tmp_path: Path) -> None:
    """A truncated or tampered file must never survive as a plausible dataset."""
    dest = tmp_path / "raw"
    with pytest.raises(ChecksumMismatchError):
        fetch(make_source(remote_file, sha256="0" * 64), dest_dir=dest)

    assert list(dest.iterdir()) == []


def test_unpinned_source_is_accepted_and_reports_its_digest(
    remote_file: Path, tmp_path: Path
) -> None:
    """Trust on first use: the first download is what establishes the checksum."""
    result = fetch(make_source(remote_file, sha256=None), dest_dir=tmp_path / "raw")

    assert result.sha256 == PAYLOAD_SHA256


def test_force_redownloads_over_an_existing_copy(remote_file: Path, tmp_path: Path) -> None:
    source = make_source(remote_file, sha256=None)
    dest = tmp_path / "raw"
    fetch(source, dest_dir=dest)
    remote_file.write_bytes(b"changed\n")

    result = fetch(source, dest_dir=dest, force=True)

    assert result.downloaded is True
    assert result.path.read_bytes() == b"changed\n"


@pytest.mark.network
@pytest.mark.parametrize("source", sources.REGISTRY.values(), ids=lambda s: s.name)
def test_registered_sources_are_still_reachable(source: sources.DataSource, tmp_path: Path) -> None:
    """Run on a schedule: pinning what to download does not guarantee it still exists."""
    result = fetch(source, dest_dir=tmp_path / "raw")

    assert result.path.stat().st_size > 0
