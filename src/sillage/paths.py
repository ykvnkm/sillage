"""Filesystem locations used across the project.

Data lives outside the package: it is never importable, never installed and never
committed. Its location is therefore resolved at runtime rather than relative to
the module, so the same code works from a git checkout, from an installed wheel
and from a Colab notebook.
"""

from __future__ import annotations

import os
from pathlib import Path

DATA_DIR_ENV = "SILLAGE_DATA_DIR"

# A checkout is recognised by these; both are absent from an installed wheel.
_ROOT_MARKERS = ("pyproject.toml", ".git")


def project_root(start: Path | None = None) -> Path:
    """Return the repository root, found by walking upwards from *start*.

    Raises:
        RuntimeError: when no marker is found, which means the code is not running
            from a checkout. Set the ``SILLAGE_DATA_DIR`` environment variable to
            say where data should live instead.
    """
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if any((candidate / marker).exists() for marker in _ROOT_MARKERS):
            return candidate
    raise RuntimeError(
        f"No project root above {current}. Running outside a checkout is fine, "
        f"but then ${DATA_DIR_ENV} must point at a data directory."
    )


def data_dir() -> Path:
    """Root of the local data tree. Overridable via ``$SILLAGE_DATA_DIR``."""
    override = os.environ.get(DATA_DIR_ENV)
    if override:
        return Path(override).expanduser().resolve()
    return project_root() / "data"


def raw_data_dir() -> Path:
    """Downloads land here, byte-identical to their source. Never modified in place."""
    return data_dir() / "raw"


MODEL_DIR_ENV = "SILLAGE_MODEL_DIR"
DEFAULT_MODEL = "odor-ensemble-v3"


def model_dir() -> Path:
    """The artifact the service loads. Overridable via ``$SILLAGE_MODEL_DIR``.

    The override is what a container uses: there is no checkout inside it, and the
    weights are mounted or downloaded to a path of the deployment's choosing.
    """
    override = os.environ.get(MODEL_DIR_ENV)
    if override:
        return Path(override).expanduser().resolve()
    return project_root() / "models" / DEFAULT_MODEL
