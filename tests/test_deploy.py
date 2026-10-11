"""The deployment scripts: checksums are enforced, and the Space gets what the image needs.

These scripts run where mistakes are expensive -- inside a Docker build and against a
public URL -- so their logic is tested here, offline, where mistakes are cheap.
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
publish_model = load("publish_model")
deploy_space = load("deploy_space")


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


@pytest.mark.parametrize(
    ("space", "url"),
    [
        ("ykvnkm/sillage", "https://ykvnkm-sillage.hf.space"),
        ("Some_User/My.Space", "https://some-user-my-space.hf.space"),
    ],
)
def test_space_url_follows_hugging_face_naming(space: str, url: str) -> None:
    assert publish_model.space_url(space) == url


def test_readme_demo_block_is_replaced_not_duplicated(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("intro\n<!-- BEGIN DEMO -->\nold\n<!-- END DEMO -->\nrest\n", "utf-8")
    manifest = {"space_url": "https://a-sillage.hf.space", "model": {"repo": "a/sillage-m"}}

    publish_model.update_readme(readme, manifest)
    publish_model.update_readme(readme, manifest)

    text = readme.read_text("utf-8")
    assert text.count("BEGIN DEMO") == 1
    assert "old" not in text
    assert "https://a-sillage.hf.space" in text


def test_space_receives_exactly_what_the_docker_build_may_see() -> None:
    """The Space builds the same Dockerfile; if the upload and the .dockerignore
    allow-list drift apart, the Space build fails where CI's did not."""
    allowed = {
        line.lstrip("!").rstrip("/")
        for line in (ROOT / ".dockerignore").read_text("utf-8").splitlines()
        if line.startswith("!")
    }
    shipped = set(deploy_space.SHIPPED) - {"Dockerfile", ".dockerignore"}

    assert shipped == allowed - {"README.md"}
