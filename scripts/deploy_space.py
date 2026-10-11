"""Deploy the service to its Hugging Face Space.

    HF_TOKEN=... uv run python scripts/deploy_space.py

Run by CI after a merge to ``main``, once tests and the image smoke test have passed.
Uploads what the Docker build needs and nothing else: the Space builds the very same
Dockerfile CI just tested, and that build downloads the model pinned in
``deploy/hub.json``. The Space card replaces README.md there; the project README
stays on GitHub.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]

# Paths the image is built from. Kept in step with the .dockerignore allow-list.
SHIPPED = (
    "Dockerfile",
    ".dockerignore",
    "pyproject.toml",
    "uv.lock",
    "src",
    "scripts/fetch_model.py",
    "deploy/hub.json",
)


def main() -> int:
    manifest = json.loads((ROOT / "deploy" / "hub.json").read_text("utf-8"))
    space_id = manifest["space"]
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(space_id, repo_type="space", space_sdk="docker", exist_ok=True)

    revision = git("rev-parse", "--short", "HEAD")
    with tempfile.TemporaryDirectory() as staging_dir:
        staging = Path(staging_dir)
        for path in tracked(SHIPPED):
            target = staging / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / path, target)
        shutil.copy2(ROOT / "deploy" / "space" / "README.md", staging / "README.md")

        api.upload_folder(
            repo_id=space_id,
            repo_type="space",
            folder_path=staging,
            commit_message=f"Deploy {revision}",
            # Files removed from the project must disappear from the Space too.
            delete_patterns=["src/**", "scripts/**", "deploy/**"],
        )

    print(f"deployed {revision} to https://huggingface.co/spaces/{space_id}")
    print(f"it will answer at {manifest['space_url']} once the Space has built")
    return 0


def tracked(paths: tuple[str, ...]) -> list[str]:
    """Only files git tracks: no caches, no local models, no stray editor files."""
    return git("ls-files", "--", *paths).splitlines()


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())
