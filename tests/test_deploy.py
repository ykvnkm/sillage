"""Fetching the published model: checksums are enforced, revisions are pinned.

The fetch script runs where mistakes are expensive -- inside a Docker build, before
any dependency exists -- so its logic is tested here, offline, where they are cheap.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(script: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(script, ROOT / "scripts" / f"{script}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fetch_model = load("fetch_model")


@pytest.fixture
def manifest(tmp_path: Path) -> Path:
    path = tmp_path / "hub.json"
    payload = {
        "model": {
            "name": "m",
            "repo": "someone/sillage-m",
            "revision": "0" * 40,
            "files": {"card.json": hashlib.sha256(b"genuine").hexdigest()},
        }
    }
    path.write_text(json.dumps(payload), "utf-8")
    return path


def test_fetch_keeps_a_file_that_matches_its_checksum(
    manifest: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    urls: list[str] = []

    def fake_download(url: str, target: Path) -> None:
        urls.append(url)
        target.write_bytes(b"genuine")

    monkeypatch.setattr(fetch_model, "download", fake_download)
    monkeypatch.setattr(
        "sys.argv", ["x", "--manifest", str(manifest), "--out", str(tmp_path / "o")]
    )

    assert fetch_model.main() == 0
    assert (tmp_path / "o" / "card.json").read_bytes() == b"genuine"
    # The pinned revision, never a branch name.
    assert urls == [f"https://huggingface.co/someone/sillage-m/resolve/{'0' * 40}/card.json"]


def test_fetch_deletes_a_file_that_does_not_match(
    manifest: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pickle that is not the one we published is a program we did not write."""
    monkeypatch.setattr(fetch_model, "download", lambda url, target: target.write_bytes(b"evil"))
    monkeypatch.setattr(
        "sys.argv", ["x", "--manifest", str(manifest), "--out", str(tmp_path / "o")]
    )

    assert fetch_model.main() == 1
    assert not (tmp_path / "o" / "card.json").exists()
