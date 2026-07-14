"""Audit local PBP coverage for the four-season target corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nhl_ml.season_scale_coverage import (
    SeasonScaleCoverageError,
    audit_season_scale_coverage,
    result_as_dict,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit local PBP files against the frozen four-season regular-season manifest."
        )
    )
    parser.add_argument(
        "--downloads-dir",
        action="append",
        type=Path,
        dest="downloads_dirs",
        help=("Directory containing nhl_pbp_<game_id>.json. May be supplied multiple times."),
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail unless every expected game is valid.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    downloads_dirs = (
        args.downloads_dirs
        if args.downloads_dirs
        else [Path.home() / "Downloads/nhl_pbp_season_scale"]
    )

    try:
        result = audit_season_scale_coverage(
            manifest_path=(
                PROJECT_ROOT / "data/interim/manifests/season_scale_regular_season.jsonl"
            ),
            manifest_audit_path=(
                PROJECT_ROOT / "storage/audits/local/season_scale_regular_season_manifest.json"
            ),
            downloads_dirs=downloads_dirs,
            audit_path=(PROJECT_ROOT / "storage/audits/local/season_scale_pbp_coverage.json"),
            missing_ids_path=(
                PROJECT_ROOT / "storage/audits/local/season_scale_pbp_missing_ids.txt"
            ),
            require_complete=args.require_complete,
        )
    except (
        FileNotFoundError,
        SeasonScaleCoverageError,
    ) as exc:
        print(f"Season-scale coverage audit failed: {exc}")
        return 1

    print(
        json.dumps(
            result_as_dict(result),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
