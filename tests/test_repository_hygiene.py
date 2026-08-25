"""Guards against the repository silently losing files.

The failure this exists for: `.gitignore` had an unanchored `models/` rule, meant
for a top-level directory of weights. Unanchored patterns match at any depth, so it
also matched `src/sillage/models/` and quietly excluded the entire model package.
`git add -A` reported nothing; the code simply never entered a commit.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from sillage.paths import project_root

PACKAGE_ROOT = "src/sillage"


def source_files() -> list[Path]:
    root = project_root()
    return sorted(
        path for path in (root / PACKAGE_ROOT).rglob("*.py") if "__pycache__" not in path.parts
    )


@pytest.mark.parametrize("path", source_files(), ids=lambda p: str(p.relative_to(project_root())))
def test_no_source_file_is_ignored_by_git(path: Path) -> None:
    """`git check-ignore` exits 0 when a path is ignored -- which no source file may be."""
    result = subprocess.run(
        ["git", "check-ignore", "-v", str(path)],
        cwd=project_root(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0, (
        f"{path.relative_to(project_root())} is ignored by git:\n  {result.stdout.strip()}"
    )
