"""Import and audit one manually saved NHL play-by-play file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nhl_ml.pbp_raw import (
    PbpRawError,
    process_manual_pbp_file,
    result_as_dict,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Import and structurally audit one manually saved NHL play-by-play JSON file.")
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to the manually saved JSON file.",
    )
    parser.add_argument(
        "--game-id",
        required=True,
        help="Expected NHL game ID.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        result = process_manual_pbp_file(
            input_path=args.input,
            expected_game_id=args.game_id,
            raw_root=PROJECT_ROOT / "storage/raw",
            manifest_path=(PROJECT_ROOT / "storage/manifests/local/nhl_web_imports.jsonl"),
            audit_path=(PROJECT_ROOT / f"storage/audits/local/pbp_{args.game_id}.json"),
        )
    except (FileNotFoundError, PbpRawError) as exc:
        print(f"PBP processing failed: {exc}")
        return 1

    print(json.dumps(result_as_dict(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
