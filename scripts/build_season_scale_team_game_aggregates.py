"""Build team-game aggregates for one verified NHL season."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nhl_ml.season_scale_pbp_batches import (
    SeasonScalePbpBatchError,
)
from nhl_ml.season_scale_team_game_aggregates import (
    build_season_scale_team_game_aggregates,
)
from nhl_ml.team_game_aggregates import (
    TeamGameAggregateError,
    result_as_dict,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ALLOWED_SEASONS = (
    "20212022",
    "20222023",
    "20232024",
    "20242025",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Build team-game aggregates for one verified NHL season.")
    )
    parser.add_argument(
        "--season-id",
        required=True,
        choices=ALLOWED_SEASONS,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        result = build_season_scale_team_game_aggregates(
            season_id=args.season_id,
            config_path=(PROJECT_ROOT / "configs" / (f"pbp_season_{args.season_id}.yaml")),
            inventory_path=(
                PROJECT_ROOT / "data/interim/inventory/season_scale_verified_games.jsonl"
            ),
            inventory_audit_path=(
                PROJECT_ROOT / "storage/audits/local/season_scale_verified_inventory.json"
            ),
            canonical_output_root=(PROJECT_ROOT / "data/interim/pbp"),
            per_game_audit_root=(
                PROJECT_ROOT / "storage/audits/local/season_scale" / args.season_id
            ),
            output_path=(
                PROJECT_ROOT / "data/interim/team_game/"
                "season_scale" / (f"team_game_{args.season_id}.jsonl")
            ),
            audit_path=(
                PROJECT_ROOT / "storage/audits/local/"
                "season_scale" / (f"team_game_{args.season_id}.json")
            ),
        )
    except (
        FileNotFoundError,
        SeasonScalePbpBatchError,
        TeamGameAggregateError,
    ) as exc:
        print(f"Season team-game build failed: {exc}")
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
