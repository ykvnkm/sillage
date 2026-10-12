"""``sillage-predict``: ask the model about a molecule from the terminal.

    uv run sillage-predict "COc1cc(C=O)ccc1O"
    uv run sillage-predict "CC1=CC[C@@H](CC1=O)C(C)=C" --json

Exit code 0 for an answer, 2 for a refusal -- a refusal is a designed outcome, but a
script calling this should be able to tell the two apart without parsing text.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from sillage.domain import Caveat, OdorReport, Refusal
from sillage.interfaces.api.schemas import result_out
from sillage.interfaces.composition import build_predict_odor

CAVEAT_TEXT = {
    Caveat.STEREO_DISCARDED: (
        "stereochemistry in the input was discarded: the model cannot tell mirror "
        "isomers apart (carvone is caraway on one side and mint on the other)"
    ),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Predict the odour of a molecule.")
    parser.add_argument("query", help="a molecule name (Russian or English) or a SMILES string")
    parser.add_argument("--neighbours", type=int, default=5, help="known molecules to show")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    result = build_predict_odor()(args.query, neighbours=args.neighbours)

    if args.json:
        print(json.dumps(to_dict(result), indent=2, ensure_ascii=False))
    else:
        print(render(result))
    return 2 if isinstance(result, Refusal) else 0


def render(result: OdorReport | Refusal) -> str:
    if isinstance(result, Refusal):
        return f"refused ({result.reason}): {result.detail}"

    molecule = result.molecule
    names = result.names
    lines = [
        f"name       {names.display}" + (f" / {names.en}" if names.ru and names.en else "")
        if names
        else "name       (not known)",
        f"molecule   {molecule.smiles}   {molecule.formula}, {molecule.weight:.1f} g/mol",
        f"model      {result.model_name}",
        "",
        "predicted",
    ]
    confident = result.profile.confident()
    if not confident:
        lines.append("  nothing at 'possible' or above")
    for score in confident:
        lines.append(
            f"  {score.confidence:<9} {score.name:<15} {score.probability:.2f}"
            f"   (occurs in {score.base_rate:.0%} of molecules)"
        )

    unverified = result.profile.unverified()
    if unverified:
        lines += ["", "leaning towards, unverified (too few examples to calibrate, no number)"]
        lines += [f"  {score.name}" for score in unverified]

    lines += ["", "experts on this exact molecule"]
    if result.known is None:
        lines.append("  not in the dataset")
    else:
        lines.append("  " + (", ".join(result.known.descriptors) or "(no descriptors)"))
        if result.known.seen_in_training:
            lines.append("  (the model was trained on this molecule: agreement here is recall)")

    lines += ["", "nearest known molecules"]
    for neighbour in result.neighbours:
        label = neighbour.names.display if neighbour.names else neighbour.smiles
        lines.append(f"  {neighbour.similarity:.2f}  {label:<32} ")
        lines[-1] += ", ".join(neighbour.descriptors[:6])

    if result.caveats:
        lines += ["", "caveats"]
        lines += [f"  ! {CAVEAT_TEXT[caveat]}" for caveat in result.caveats]
    return "\n".join(lines)


def to_dict(result: OdorReport | Refusal) -> dict[str, Any]:
    """The same JSON the HTTP API returns: one contract, two entry points."""
    return result_out(result).model_dump()


if __name__ == "__main__":
    raise SystemExit(main())
