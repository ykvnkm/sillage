"""Molecule drawings as SVG, rendered by RDKit on the server.

Server-side on purpose: the browser gets a finished vector image and needs no
chemistry library of its own. One system dependency comes with it: RDKit's drawing
module links against the X11 libraries ``libXrender`` and ``libXext`` even when it
only writes SVG, so the container installs them (see the Dockerfile).

Drawn in one ink colour rather than RDKit's element palette. On a page that uses
colour to mean confidence, a red oxygen would compete with it; the element letters
already say which atom is which. The one other use of colour is shading atoms by how
much they carry a note (journal E-021): amber for, blue against.
"""

from __future__ import annotations

from functools import lru_cache

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from sillage.domain import InvalidMolecule

INK = (0.149, 0.098, 0.059)  # the page's ink, #26190f
MIN_SIDE, MAX_SIDE = 80, 800


CARRIES = (0.816, 0.541, 0.196)  # --amber-400, #d08a32: the atom carries the note
HOLDS_BACK = (0.310, 0.486, 0.675)  # #4f7cac: the atom holds the note back
FULL_SHADE_AT = 0.15
"""Probability points at or above which the strongest atom of a molecule is shaded at
full strength; the rest fade towards the paper in proportion. A molecule whose atoms
all matter little stays pale instead of being stretched to look decisive."""


class RdkitDepicter:
    def svg(
        self, smiles: str, width: int, height: int, weights: tuple[float, ...] | None = None
    ) -> str:
        width = _clamp(width)
        height = _clamp(height)
        return _draw(smiles, width, height, tuple(weights) if weights is not None else None)


def _clamp(side: int) -> int:
    return max(MIN_SIDE, min(MAX_SIDE, int(side)))


@lru_cache(maxsize=2048)
def _draw(smiles: str, width: int, height: int, weights: tuple[float, ...] | None) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise InvalidMolecule(f"not a valid SMILES string: {smiles!r}")
    if weights is not None and len(weights) != molecule.GetNumAtoms():
        raise InvalidMolecule(
            f"{len(weights)} weights for {molecule.GetNumAtoms()} atoms of {smiles!r}"
        )

    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    options = drawer.drawOptions()
    options.clearBackground = False
    options.padding = 0.08
    options.bondLineWidth = 1.6
    options.minFontSize = 11
    options.updateAtomPalette({atomic_number: INK for atomic_number in range(119)})
    options.setSymbolColour(INK)

    highlight = _shades(weights) if weights is not None else {}
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        molecule,
        highlightAtoms=list(highlight),
        highlightAtomColors=highlight,
        highlightAtomRadii={atom: 0.42 for atom in highlight},
        highlightBonds=[],
    )
    drawer.FinishDrawing()
    # RDKit emits an XML declaration and fixed pixel size; the page sizes the image
    # itself, so keep the viewBox and drop the rest.
    svg = drawer.GetDrawingText()
    svg = svg.replace("<?xml version='1.0' encoding='iso-8859-1'?>\n", "")
    return svg.replace(f"width='{width}px' height='{height}px'", "", 1)


def _shades(weights: tuple[float, ...]) -> dict[int, tuple[float, float, float]]:
    """Each atom's colour, blended towards the paper by how little it matters."""
    paper = (0.992, 0.988, 0.976)
    full = max([FULL_SHADE_AT, *(abs(w) for w in weights)])
    shades = {}
    for atom, weight in enumerate(weights):
        strength = abs(weight) / full
        if strength < 1 / 15:
            continue
        hue = CARRIES if weight > 0 else HOLDS_BACK
        shades[atom] = tuple(p + (h - p) * strength for p, h in zip(paper, hue, strict=True))
    return shades
