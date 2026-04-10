from __future__ import annotations

import argparse
from pathlib import Path
import sys

from stub_gguf.generate import generate_stub_gguf
from stub_gguf.validate import validate_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stub-gguf")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument(
        "--output",
        type=Path,
        default=Path("dist/noted.gguf"),
    )

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument(
        "--input",
        type=Path,
        default=Path("dist/noted.gguf"),
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "generate":
        written = generate_stub_gguf(args.output)
        print(f"Wrote {written}")
    elif args.command == "validate":
        try:
            validate_file(args.input)
        except (OSError, ValueError) as error:
            print(f"Validation failed: {error}", file=sys.stderr)
            return 1
        print(f"Validated {args.input}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
