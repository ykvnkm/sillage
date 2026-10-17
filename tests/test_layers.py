"""The dependency rule of ADR-0001, enforced rather than agreed on.

``domain`` and ``application`` must import without a single third-party package.
Checked in a fresh interpreter: inside pytest, RDKit and NumPy are already loaded by
other tests, so ``sys.modules`` would prove nothing.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

HEAVY = ("rdkit", "numpy", "sklearn", "pandas", "pandera", "joblib", "scipy", "torch")


@pytest.mark.parametrize("package", ["sillage.domain", "sillage.application"])
def test_inner_layers_import_no_third_party_package(package: str) -> None:
    probe = (
        f"import sys, {package}\n"
        f"heavy = {HEAVY!r}\n"
        "print(sorted({m.split('.')[0] for m in sys.modules} & set(heavy)))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "[]", (
        f"{package} pulled in {result.stdout.strip()}: the dependency rule is broken"
    )


@pytest.mark.parametrize(
    "module", ["sillage.interfaces.api.app", "sillage.interfaces.cli", "sillage.features.graphs"]
)
def test_the_service_never_imports_torch(module: str) -> None:
    """Torch is a training dependency (the ``gnn`` group); the image does not have it."""
    probe = f"import sys, {module}\nprint('torch' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False", f"{module} imports torch"
