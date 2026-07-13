"""Canonicalize one manually imported NHL daily-score raw file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nhl_ml.canonical.nhl_games import (
    NHLCanonicalizationError,
    canonicalize_score_file,
    result_as_dict,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Canonicalize one previously imported NHL daily-score raw file.")
    )
    parser.add_argument(
        "--raw-file",
        required=True,
        type=Path,
        help="Path to one raw NHL daily-score JSON file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        result = canonicalize_score_file(
            raw_file=args.raw_file,
            raw_root=PROJECT_ROOT / "storage/raw",
            import_manifest_path=(PROJECT_ROOT / "storage/manifests/local/nhl_web_imports.jsonl"),
            output_root=PROJECT_ROOT / "data/interim",
        )
    except (
        FileNotFoundError,
        NHLCanonicalizationError,
    ) as exc:
        print(f"Canonicalization failed: {exc}")
        return 1

    print(json.dumps(result_as_dict(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
