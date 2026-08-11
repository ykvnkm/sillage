"""Fetching sources from the registry, with integrity verification.

Two properties matter more than speed here.

*Idempotence*: a second run must not hit the network. A local copy whose digest
matches is accepted as-is, which makes the fetch command safe to put in setup
instructions and in CI.

*Atomicity*: bytes are written to a temporary ``.part`` file and only renamed into
place after the digest checks out. An interrupted download therefore cannot leave
a truncated file that looks valid -- the failure mode that silently trains a model
on half a dataset.
"""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from sillage.data.sources import DataSource
from sillage.paths import raw_data_dir

_CHUNK_BYTES = 1 << 20
_TIMEOUT_SECONDS = 60


class ChecksumMismatchError(RuntimeError):
    """Raised when downloaded bytes do not match the digest pinned in the registry."""

    def __init__(self, filename: str, expected: str, actual: str) -> None:
        super().__init__(
            f"{filename}: expected sha256 {expected}, got {actual}. "
            "Either the source changed under us, the download is truncated, "
            "or the file was tampered with. Investigate before re-pinning."
        )
        self.filename = filename
        self.expected = expected
        self.actual = actual


@dataclass(frozen=True, slots=True)
class FetchResult:
    """Outcome of a fetch, including the digest actually observed on disk."""

    source: DataSource
    path: Path
    sha256: str
    downloaded: bool
    """False when a valid local copy was reused and no network access happened."""


def sha256_of(path: Path) -> str:
    """Digest a file in chunks, so that arbitrarily large datasets fit in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(source: DataSource, *, dest_dir: Path | None = None, force: bool = False) -> FetchResult:
    """Download *source* unless a valid copy is already present.

    Args:
        source: registry entry to fetch.
        dest_dir: where to place the file; defaults to ``data/raw``.
        force: re-download even when a valid local copy exists.

    Raises:
        ChecksumMismatchError: when the digest does not match a pinned checksum.
    """
    directory = dest_dir if dest_dir is not None else raw_data_dir()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / source.filename

    if target.exists() and not force:
        digest = sha256_of(target)
        _verify(source, target.name, digest)
        return FetchResult(source=source, path=target, sha256=digest, downloaded=False)

    partial = target.with_name(target.name + ".part")
    try:
        _stream_to_file(source.url, partial)
        digest = sha256_of(partial)
        _verify(source, target.name, digest)
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)

    return FetchResult(source=source, path=target, sha256=digest, downloaded=True)


def _verify(source: DataSource, filename: str, digest: str) -> None:
    """Compare against the pinned digest. An unpinned source cannot be verified."""
    if source.sha256 is not None and digest != source.sha256:
        raise ChecksumMismatchError(filename, source.sha256, digest)


def _stream_to_file(url: str, destination: Path) -> None:
    """Copy a URL to disk without holding the whole payload in memory."""
    request = urllib.request.Request(url, headers={"User-Agent": "sillage-data"})
    with (
        urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response,
        destination.open("wb") as handle,
    ):
        shutil.copyfileobj(response, handle, _CHUNK_BYTES)
