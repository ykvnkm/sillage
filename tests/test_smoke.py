"""Smoke tests: the package is installed and the toolchain is wired up correctly."""

from importlib.metadata import version

import sillage


def test_package_is_properly_installed() -> None:
    """Import must resolve to an installed distribution, not a stray source directory.

    Under the src/ layout, ``import sillage`` can only succeed if the package was
    really installed into the environment. Comparing the runtime ``__version__``
    against the distribution metadata additionally proves that the build backend
    read the version from the single source of truth.
    """
    assert version("sillage") == sillage.__version__
