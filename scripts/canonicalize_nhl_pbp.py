"""Canonicalize one imported NHL play-by-play file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nhl_ml.pbp_canonical import (
    PbpCanonicalizationError,
    canonicalize_pbp_file,
    result_as_dict,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Canonicalize one imported NHL PBP raw file.")
    parser.add_argument(
        "--raw-file",
        required=True,
        type=Path,
        help="Path to the imported raw PBP JSON file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    raw_file = args.raw_file.expanduser().resolve()

    try:
        game_id = raw_file.parent.name

        result = canonicalize_pbp_file(
            raw_file=raw_file,
            raw_root=PROJECT_ROOT / "storage/raw",
            import_manifest_path=(PROJECT_ROOT / "storage/manifests/local/nhl_web_imports.jsonl"),
            output_root=PROJECT_ROOT / "data/interim/pbp",
            audit_path=(PROJECT_ROOT / f"storage/audits/local/pbp_canonical_{game_id}.json"),
        )
    except (
        FileNotFoundError,
        PbpCanonicalizationError,
    ) as exc:
        print(f"PBP canonicalization failed: {exc}")
        return 1

    print(json.dumps(result_as_dict(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
