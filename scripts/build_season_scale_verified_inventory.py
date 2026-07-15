"""Build the verified four-season NHL regular-season inventory."""

from __future__ import annotations

import json
from pathlib import Path

from nhl_ml.season_scale_inventory import (
    SeasonScaleInventoryError,
    build_verified_season_scale_inventory,
    result_as_dict,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    try:
        result = build_verified_season_scale_inventory(
            manifest_path=(
                PROJECT_ROOT / "data/interim/manifests/season_scale_regular_season.jsonl"
            ),
            manifest_audit_path=(
                PROJECT_ROOT / "storage/audits/local/season_scale_regular_season_manifest.json"
            ),
            downloads_dir=(Path.home() / "Downloads/nhl_pbp_season_scale"),
            output_path=(PROJECT_ROOT / "data/interim/inventory/season_scale_verified_games.jsonl"),
            audit_path=(PROJECT_ROOT / "storage/audits/local/season_scale_verified_inventory.json"),
        )
    except (
        FileNotFoundError,
        SeasonScaleInventoryError,
    ) as exc:
        print(f"Verified inventory build failed: {exc}")
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
