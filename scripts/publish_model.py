"""Publish a trained model to the Hugging Face Hub and pin it for the Docker image.

    hf auth login                                  # once, with a write token
    uv run python scripts/publish_model.py

Uploads ``models/<name>/`` with its model card to ``<you>/sillage-<name>``, then
writes ``deploy/hub.json``: the repository, the exact commit it was uploaded as, and
the sha256 of every file. That manifest is committed to git. The Docker build
downloads exactly that revision and refuses any file whose checksum differs: the
weights are pickles, and unpickling executes code, so an unverified file is not a
model but an arbitrary program.

The account name comes from the token, so nothing here is specific to one person.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

from sillage.models.artifact import (
    BOOSTING_FILE,
    CALIBRATOR_FILE,
    CARD_FILE,
    ESTIMATOR_FILE,
    GRAPH_FILE,
    REFERENCE_FILE,
    OdorModel,
)
from sillage.paths import DEFAULT_MODEL, project_root

ARTIFACT_FILES = (CARD_FILE, ESTIMATOR_FILE, CALIBRATOR_FILE, REFERENCE_FILE)
NAMES_FILE = "names.json"
MAP_FILE = "map.npz"
OPTIONAL_FILES = (BOOSTING_FILE, GRAPH_FILE, NAMES_FILE, MAP_FILE)
"""The ensemble's other components (journal E-020), then what scripts/build_names.py
and scripts/build_map.py add. Published when present."""
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
[sillage]({github}), which also serves it through a web interface and an HTTP API.

Load it with the project code (`OdorModel.load`), not with a bare `joblib.load`:
the artifact carries its feature sets, components, calibrator and chemical scope,
and outputs are only meaningful together with them. The `.joblib` files are
scikit-learn pickles; load them only from a revision you trust, and check them
against the sha256 values pinned in the project's `deploy/hub.json`. `graph.npz`
holds plain arrays and is read without pickle.

Data: the model learnt from the Goodscents and Leffingwell collections as archived
by [Pyrfume](https://pyrfume.org); their records are not redistributed here, only
weights derived from them. `reference.json` comes from the merge curated by
[OpenPOM](https://github.com/BioMachineLearning/openpom), MIT. Molecule names in
`names.json`: [PubChem](https://pubchem.ncbi.nlm.nih.gov/) (public domain) and
[Wikidata](https://www.wikidata.org/) (CC0).

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
    # surface when the Docker image is built from the Hub.
    model = OdorModel.load(artifact)
    print(f"loaded {args.name}: {len(model.label_names)} labels, {len(model.reference)} references")

    api = HfApi()
    user = api.whoami()["name"]
    repo_id = f"{user}/sillage-{args.name}"

    api.create_repo(repo_id, repo_type="model", exist_ok=True)

    files = [*ARTIFACT_FILES, *(f for f in OPTIONAL_FILES if (artifact / f).exists())]

    with tempfile.TemporaryDirectory() as staging_dir:
        staging = Path(staging_dir)
        for name in files:
            shutil.copy2(artifact / name, staging / name)
        readme = FRONT_MATTER.format(name=args.name, github=GITHUB_URL)
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
            "files": {name: sha256(artifact / name) for name in files},
        },
    }
    manifest_path = root / "deploy" / "hub.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", "utf-8")

    print(f"published https://huggingface.co/{repo_id} at {commit.oid[:12]}")
    print(f"wrote {manifest_path.relative_to(root)}: commit it so the image builds this revision")
    return 0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
