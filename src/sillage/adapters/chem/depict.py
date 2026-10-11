"""Molecule drawings as SVG, rendered by RDKit on the server.

Server-side on purpose: the browser gets a finished vector image and needs no
chemistry library of its own. One system dependency comes with it: RDKit's drawing
module links against the X11 libraries ``libXrender`` and ``libXext`` even when it
only writes SVG, so the container installs them (see the Dockerfile).

Drawn in one ink colour rather than RDKit's element palette. On a page that uses
colour to mean confidence, a red oxygen would compete with it; the element letters
already say which atom is which.
"""

from __future__ import annotations

from functools import lru_cache

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from sillage.domain import InvalidMolecule

INK = (0.149, 0.098, 0.059)  # the page's ink, #26190f
MIN_SIDE, MAX_SIDE = 80, 800


class RdkitDepicter:
    def svg(self, smiles: str, width: int, height: int) -> str:
        width = _clamp(width)
        height = _clamp(height)
        return _draw(smiles, width, height)


def _clamp(side: int) -> int:
    return max(MIN_SIDE, min(MAX_SIDE, int(side)))


@lru_cache(maxsize=2048)
def _draw(smiles: str, width: int, height: int) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise InvalidMolecule(f"not a valid SMILES string: {smiles!r}")

    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    options = drawer.drawOptions()
    options.clearBackground = False
    options.padding = 0.08
    options.bondLineWidth = 1.6
    options.minFontSize = 11
    options.updateAtomPalette({atomic_number: INK for atomic_number in range(119)})
    options.setSymbolColour(INK)

    rdMolDraw2D.PrepareAndDrawMolecule(drawer, molecule)
    drawer.FinishDrawing()
    # RDKit emits an XML declaration and fixed pixel size; the page sizes the image
    # itself, so keep the viewBox and drop the rest.
    svg = drawer.GetDrawingText()
    svg = svg.replace("<?xml version='1.0' encoding='iso-8859-1'?>\n", "")
    return svg.replace(f"width='{width}px' height='{height}px'", "", 1)
