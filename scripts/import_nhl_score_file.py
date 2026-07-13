"""Import one manually saved NHL daily-score JSON file."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from nhl_ml.sources.nhl_web import (
    NHLRawImportError,
    import_manual_score_file,
    result_as_dict,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Import one manually saved NHL daily-score JSON file. "
            "This command performs no network access."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to the manually saved JSON file.",
    )
    parser.add_argument(
        "--date",
        required=True,
        type=date.fromisoformat,
        help="Requested NHL score date in YYYY-MM-DD format.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        result = import_manual_score_file(
            input_path=args.input,
            requested_date=args.date,
            raw_root=PROJECT_ROOT / "storage/raw",
            manifest_path=(PROJECT_ROOT / "storage/manifests/local/nhl_web_imports.jsonl"),
        )
    except (FileNotFoundError, NHLRawImportError) as exc:
        print(f"Import failed: {exc}")
        return 1

    print(json.dumps(result_as_dict(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
