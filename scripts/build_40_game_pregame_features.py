"""Build leakage-safe pre-game features for the forty-game pilot."""

from __future__ import annotations

import json
from pathlib import Path

from nhl_ml.rolling_features import (
    RollingFeatureError,
    build_pregame_rolling_features,
    result_as_dict,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    try:
        result = build_pregame_rolling_features(
            team_game_path=(PROJECT_ROOT / "data/interim/team_game/multiseason_40_team_game.jsonl"),
            team_game_audit_path=(
                PROJECT_ROOT / "storage/audits/local/multiseason_40_team_game.json"
            ),
            output_path=(
                PROJECT_ROOT / "data/interim/features/multiseason_40_pregame_team_features.jsonl"
            ),
            audit_path=(
                PROJECT_ROOT / "storage/audits/local/multiseason_40_pregame_team_features.json"
            ),
        )
    except (
        FileNotFoundError,
        RollingFeatureError,
    ) as exc:
        print(f"Pre-game feature build failed: {exc}")
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
