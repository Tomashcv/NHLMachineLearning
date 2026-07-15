"""Process one verified season-scale NHL PBP batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nhl_ml.config import ConfigError
from nhl_ml.pbp_batch import (
    PbpBatchConfig,
    PbpBatchError,
    result_as_dict,
    run_pbp_batch,
)
from nhl_ml.pbp_canonical import PbpCanonicalizationError
from nhl_ml.pbp_raw import PbpRawError
from nhl_ml.season_scale_pbp_batches import (
    SeasonScalePbpBatchError,
    verify_season_pbp_batch_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ALLOWED_SEASONS = (
    "20212022",
    "20222023",
    "20232024",
    "20242025",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process one frozen NHL regular-season PBP batch.")
    parser.add_argument(
        "--season-id",
        required=True,
        choices=ALLOWED_SEASONS,
    )
    parser.add_argument(
        "--downloads-dir",
        type=Path,
        default=(Path.home() / "Downloads/nhl_pbp_season_scale"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = PROJECT_ROOT / "configs" / f"pbp_season_{args.season_id}.yaml"

    try:
        verification = verify_season_pbp_batch_config(
            season_id=args.season_id,
            config_path=config_path,
            inventory_path=(
                PROJECT_ROOT / "data/interim/inventory/season_scale_verified_games.jsonl"
            ),
            inventory_audit_path=(
                PROJECT_ROOT / "storage/audits/local/season_scale_verified_inventory.json"
            ),
        )

        config = PbpBatchConfig.from_path(config_path)

        result = run_pbp_batch(
            config=config,
            downloads_dir=args.downloads_dir,
            raw_root=PROJECT_ROOT / "storage/raw",
            manifest_path=(PROJECT_ROOT / "storage/manifests/local/nhl_web_imports.jsonl"),
            canonical_output_root=(PROJECT_ROOT / "data/interim/pbp"),
            per_game_audit_root=(
                PROJECT_ROOT / "storage/audits/local/season_scale" / args.season_id
            ),
            aggregate_audit_path=(
                PROJECT_ROOT
                / "storage/audits/local/season_scale"
                / f"{args.season_id}_pbp_batch.json"
            ),
        )
    except (
        ConfigError,
        FileNotFoundError,
        PbpBatchError,
        PbpCanonicalizationError,
        PbpRawError,
        SeasonScalePbpBatchError,
    ) as exc:
        print(f"Season PBP batch failed: {exc}")
        return 1

    print("Config verification:")
    print(json.dumps(verification, indent=2, sort_keys=True))
    print()
    print("Batch result:")
    print(json.dumps(result_as_dict(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
