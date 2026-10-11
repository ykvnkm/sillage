"""Download the pinned model from the Hugging Face Hub and verify every file.

    python scripts/fetch_model.py                       # -> models/<name>/
    python scripts/fetch_model.py --out /app/model      # what the Docker build runs

Standard library only, on purpose: it runs inside the Docker build before any
dependency is installed, and on a fresh checkout before ``uv sync``.

Reads ``deploy/hub.json``, written by ``publish_model.py``. Downloads the exact
revision recorded there, never a branch, and rejects any file whose sha256 differs
from the manifest. The files are pickles and unpickling runs code, so a file that
does not match is deleted rather than kept "just in case".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RETRIES = 4


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "deploy" / "hub.json")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    model = json.loads(args.manifest.read_text("utf-8"))["model"]
    out = args.out or ROOT / "models" / model["name"]
    out.mkdir(parents=True, exist_ok=True)

    for name, expected in model["files"].items():
        target = out / name
        if target.exists() and sha256(target) == expected:
            print(f"ok      {name} (already present)")
            continue
        url = f"https://huggingface.co/{model['repo']}/resolve/{model['revision']}/{name}"
        download(url, target)
        actual = sha256(target)
        if actual != expected:
            target.unlink()
            print(f"FAILED  {name}: sha256 {actual} does not match the manifest", file=sys.stderr)
            return 1
        print(f"ok      {name}")

    print(f"{model['repo']}@{model['revision'][:12]} -> {out}")
    return 0


def download(url: str, target: Path) -> None:
    for attempt in range(1, RETRIES + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                target.write_bytes(response.read())
            return
        except OSError as error:
            if attempt == RETRIES:
                raise
            print(f"retry   {target.name}: {error}", file=sys.stderr)
            time.sleep(2**attempt)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
