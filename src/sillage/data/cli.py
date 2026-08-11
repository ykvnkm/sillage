"""Command line entry point: ``uv run sillage-data``.

Kept deliberately thin. It parses arguments and prints; every decision lives in
:mod:`sillage.data.download` and :mod:`sillage.data.sources`, where it can be
tested without a subprocess.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from sillage.data import sources
from sillage.data.download import ChecksumMismatchError, fetch
from sillage.paths import raw_data_dir


def _cmd_list(_: argparse.Namespace) -> int:
    for source in sorted(sources.REGISTRY.values(), key=lambda s: s.name):
        pinned = source.sha256[:12] if source.sha256 else "NOT PINNED"
        print(f"{source.name:<28} {source.licence:<10} {pinned:<12} {source.description}")
    return 0


def _cmd_provenance(_: argparse.Namespace) -> int:
    """Emit the README table, so provenance is derived from the registry, not retyped."""
    print("| Source | Licence | Redistribution | Description |")
    print("| --- | --- | --- | --- |")
    for source in sorted(sources.REGISTRY.values(), key=lambda s: s.name):
        link = f"[{source.name}]({source.homepage})"
        print(f"| {link} | {source.licence} | {source.redistribution} | {source.description} |")
    return 0


def _cmd_fetch(args: argparse.Namespace) -> int:
    names = args.names or sorted(sources.REGISTRY)
    exit_code = 0
    for name in names:
        source = sources.get(name)
        try:
            result = fetch(source, force=args.force)
        except ChecksumMismatchError as error:
            print(f"FAIL  {name}: {error}", file=sys.stderr)
            exit_code = 1
            continue

        action = "downloaded" if result.downloaded else "cached"
        print(f"OK    {name}: {action} -> {result.path}")
        if not source.is_pinned:
            print(
                f"      This source is not pinned yet. Record the checksum in\n"
                f"      src/sillage/data/sources.py to make it verifiable:\n"
                f'          sha256="{result.sha256}",'
            )
    return exit_code


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sillage-data",
        description=f"Download external data sources into {raw_data_dir()}",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    listing = subcommands.add_parser("list", help="show every registered source")
    listing.set_defaults(handler=_cmd_list)

    provenance = subcommands.add_parser("provenance", help="print the README provenance table")
    provenance.set_defaults(handler=_cmd_provenance)

    fetching = subcommands.add_parser("fetch", help="download sources and verify checksums")
    fetching.add_argument("names", nargs="*", help="source names; default is all of them")
    fetching.add_argument("--force", action="store_true", help="re-download even if present")
    fetching.set_defaults(handler=_cmd_fetch)

    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
