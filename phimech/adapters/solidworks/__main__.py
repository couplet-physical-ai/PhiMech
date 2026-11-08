"""Portable SolidWorks handoff: deterministic STEP metadata validation only.

Native add-in, proprietary SDK, drawing standards and model-driven review are separate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .bundle import _json, package_results, prepare_bundle, sample_metadata, validate_bundle


def _handle(args: Any) -> int:
    try:
        if args.action == "validate":
            result = validate_bundle(args.bundle, require_kernel=not args.preflight,
                                     source_path=args.source)
        elif args.action == "package":
            result = package_results(args.bundle, args.drawing, args.output,
                                     review_json=args.review)
        else:
            metadata = (sample_metadata(args.step) if args.action == "sample"
                        else _json(Path(args.metadata)))
            result = prepare_bundle(args.step, metadata, args.output,
                                    require_kernel=not args.preflight, source_path=args.source)
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        print(json.dumps({"status": "refused", "error": str(exc)}))
        return 2


def _commands(parser: Any) -> None:
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("sample", "prepare", "validate", "package"):
        sub = commands.add_parser(action)
        sub.set_defaults(handler=_handle)
        if action in ("sample", "prepare"):
            sub.add_argument("step")
            sub.add_argument("-o", "--output", required=True)
            if action == "prepare":
                sub.add_argument("--metadata", required=True)
        else:
            sub.add_argument("bundle")
        if action != "package":
            sub.add_argument("--preflight", action="store_true",
                             help="metadata contract only; not geometry acceptance")
            sub.add_argument("--source", help="optional original file for source hash verification")
        else:
            sub.add_argument("--drawing", required=True)
            sub.add_argument("--review")
            sub.add_argument("-o", "--output", required=True)


def register_cli(subparsers: Any) -> None:
    _commands(subparsers.add_parser("solidworks", help="validate isolated STEP handoffs"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _commands(parser)
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
