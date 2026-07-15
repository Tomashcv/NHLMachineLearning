"""Build the leakage-safe season-scale NHL model-ready panel."""

from __future__ import annotations

import json
from pathlib import Path

from nhl_ml.model_ready_panel import (
    ModelReadyPanelError,
    build_model_ready_panel,
    result_as_dict,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    try:
        result = build_model_ready_panel(
            team_game_path=(
                PROJECT_ROOT / "data/interim/team_game/season_scale/team_game_2021_2025.jsonl"
            ),
            team_game_audit_path=(
                PROJECT_ROOT / "storage/audits/local/season_scale/team_game_2021_2025.json"
            ),
            pregame_path=(
                PROJECT_ROOT / "data/interim/features/"
                "season_scale/"
                "pregame_team_features_"
                "2021_2025.jsonl"
            ),
            pregame_audit_path=(
                PROJECT_ROOT / "storage/audits/local/"
                "season_scale/"
                "pregame_team_features_"
                "2021_2025.json"
            ),
            output_path=(
                PROJECT_ROOT / "data/interim/model_ready/"
                "season_scale/"
                "nhl_model_ready_"
                "2021_2025.jsonl"
            ),
            audit_path=(
                PROJECT_ROOT / "storage/audits/local/season_scale/nhl_model_ready_2021_2025.json"
            ),
        )
    except (
        FileNotFoundError,
        ModelReadyPanelError,
    ) as exc:
        print(f"Model-ready panel build failed: {exc}")
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
