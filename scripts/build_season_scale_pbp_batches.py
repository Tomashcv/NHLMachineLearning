"""Build the four frozen season-scale NHL PBP batch configs."""

from __future__ import annotations

import json
from pathlib import Path

from nhl_ml.season_scale_pbp_batches import (
    SeasonScalePbpBatchError,
    build_season_scale_pbp_batch_configs,
    result_as_dict,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    try:
        result = build_season_scale_pbp_batch_configs(
            inventory_path=(
                PROJECT_ROOT / "data/interim/inventory/season_scale_verified_games.jsonl"
            ),
            inventory_audit_path=(
                PROJECT_ROOT / "storage/audits/local/season_scale_verified_inventory.json"
            ),
            output_dir=PROJECT_ROOT / "configs",
        )
    except (
        FileNotFoundError,
        SeasonScalePbpBatchError,
    ) as exc:
        print(f"Season PBP config build failed: {exc}")
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
