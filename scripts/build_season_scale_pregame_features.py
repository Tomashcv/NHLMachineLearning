"""Build the combined leakage-safe NHL season-scale pregame panel."""

from __future__ import annotations

import json
from pathlib import Path

from nhl_ml.rolling_features import (
    RollingFeatureError,
    build_pregame_rolling_features,
    result_as_dict,
)
from nhl_ml.season_scale_pregame_features import (
    build_season_scale_team_game_panel,
    panel_result_as_dict,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SEASON_IDS = (
    "20212022",
    "20222023",
    "20232024",
    "20242025",
)


def main() -> int:
    combined_team_game_path = (
        PROJECT_ROOT / "data/interim/team_game/season_scale/team_game_2021_2025.jsonl"
    )
    combined_team_game_audit_path = (
        PROJECT_ROOT / "storage/audits/local/season_scale/team_game_2021_2025.json"
    )

    try:
        panel_result = build_season_scale_team_game_panel(
            season_ids=SEASON_IDS,
            team_game_root=(PROJECT_ROOT / "data/interim/team_game/season_scale"),
            source_audit_root=(PROJECT_ROOT / "storage/audits/local/season_scale"),
            output_path=(combined_team_game_path),
            audit_path=(combined_team_game_audit_path),
            expected_team_games_per_season=82,
        )

        feature_result = build_pregame_rolling_features(
            team_game_path=(combined_team_game_path),
            team_game_audit_path=(combined_team_game_audit_path),
            output_path=(
                PROJECT_ROOT / "data/interim/features/"
                "season_scale/"
                "pregame_team_features_"
                "2021_2025.jsonl"
            ),
            audit_path=(
                PROJECT_ROOT / "storage/audits/local/"
                "season_scale/"
                "pregame_team_features_"
                "2021_2025.json"
            ),
        )
    except (
        FileNotFoundError,
        RollingFeatureError,
    ) as exc:
        print(f"Season-scale pregame feature build failed: {exc}")
        return 1

    print(
        json.dumps(
            {
                "combined_team_game_panel": (panel_result_as_dict(panel_result)),
                "pregame_features": (result_as_dict(feature_result)),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
