"""Publish a trained model to the Hugging Face Hub and pin it for deployment.

    hf auth login                                  # once, with a write token
    uv run python scripts/publish_model.py

Uploads ``models/<name>/`` with its model card to ``<you>/sillage-<name>``, then
writes ``deploy/hub.json``: the repository, the exact commit it was uploaded as, and
the sha256 of every file. That manifest is committed to git. The Docker build and the
Space download exactly that revision and refuse any file whose checksum differs:
the weights are pickles, and unpickling executes code, so an unverified file is not
a model but an arbitrary program.

The account name comes from the token, so nothing here is specific to one person.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

from sillage.models.artifact import (
    CALIBRATOR_FILE,
    CARD_FILE,
    ESTIMATOR_FILE,
    REFERENCE_FILE,
    OdorModel,
)
from sillage.paths import DEFAULT_MODEL, project_root

ARTIFACT_FILES = (CARD_FILE, ESTIMATOR_FILE, CALIBRATOR_FILE, REFERENCE_FILE)
SPACE_NAME = "sillage"
GITHUB_URL = "https://github.com/ykvnkm/sillage"

FRONT_MATTER = """---
license: mit
library_name: sklearn
language:
  - ru
  - en
tags:
  - chemistry
  - olfaction
  - odor
  - smiles
  - multilabel-classification
  - calibration
---

# {name}

Predicts which of 138 odour descriptors fit a molecule, from its structure, with
per-label calibrated probabilities. Trained and served by
[sillage]({github}); try it in the [web interface]({space_url}).

Load it with the project code (`OdorModel.load`), not with a bare `joblib.load`:
the artifact carries its feature set, calibrator and chemical scope, and outputs are
only meaningful together with them. Files are scikit-learn pickles; load them only
from a revision you trust, and check them against the sha256 values pinned in the
project's `deploy/hub.json`.

Data: Goodscents/Leffingwell merge curated by
[OpenPOM](https://github.com/BioMachineLearning/openpom), MIT.

---

"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default=DEFAULT_MODEL)
    args = parser.parse_args()

    root = project_root()
    artifact = root / "models" / args.name
    card_md = root / "docs" / "model-cards" / f"{args.name}.md"

    # Refuse to publish something that does not load: a broken upload would only
    # surface when the container starts on the public Space.
    model = OdorModel.load(artifact)
    print(f"loaded {args.name}: {len(model.label_names)} labels, {len(model.reference)} references")

    api = HfApi()
    user = api.whoami()["name"]
    repo_id = f"{user}/sillage-{args.name}"
    space_id = f"{user}/{SPACE_NAME}"

    api.create_repo(repo_id, repo_type="model", exist_ok=True)

    with tempfile.TemporaryDirectory() as staging_dir:
        staging = Path(staging_dir)
        for name in ARTIFACT_FILES:
            shutil.copy2(artifact / name, staging / name)
        readme = FRONT_MATTER.format(
            name=args.name, github=GITHUB_URL, space_url=space_url(space_id)
        )
        (staging / "README.md").write_text(readme + card_md.read_text("utf-8"), "utf-8")

        commit = api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=staging,
            commit_message=f"Publish {args.name}",
        )

    manifest = {
        "model": {
            "name": args.name,
            "repo": repo_id,
            "revision": commit.oid,
            "files": {name: sha256(artifact / name) for name in ARTIFACT_FILES},
        },
        "space": space_id,
        "space_url": space_url(space_id),
    }
    manifest_path = root / "deploy" / "hub.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", "utf-8")
    update_readme(root / "README.md", manifest)

    print(f"published https://huggingface.co/{repo_id} at {commit.oid[:12]}")
    print(f"wrote {manifest_path.relative_to(root)}; the Space will be {space_url(space_id)}")
    return 0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def space_url(space_id: str) -> str:
    """``owner/name`` -> ``https://owner-name.hf.space``, as Hugging Face derives it."""
    slug = re.sub(r"[^a-z0-9]+", "-", space_id.lower().replace("/", "-")).strip("-")
    return f"https://{slug}.hf.space"


def update_readme(readme: Path, manifest: dict) -> None:
    """Keep the README's demo links in step with what is actually published."""
    block = (
        "<!-- BEGIN DEMO -->\n"
        f"**Демо:** [{manifest['space_url']}]({manifest['space_url']}) · "
        f"веса: [huggingface.co/{manifest['model']['repo']}]"
        f"(https://huggingface.co/{manifest['model']['repo']})\n"
        "<!-- END DEMO -->"
    )
    text = readme.read_text("utf-8")
    text = re.sub(r"<!-- BEGIN DEMO -->.*?<!-- END DEMO -->", block, text, flags=re.S)
    readme.write_text(text, "utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
