"""Build team-game aggregates for the frozen forty-game NHL pilot."""

from __future__ import annotations

import json
from pathlib import Path

from nhl_ml.team_game_aggregates import (
    TeamGameAggregateError,
    build_team_game_aggregates,
    result_as_dict,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    try:
        result = build_team_game_aggregates(
            pilot_path=(PROJECT_ROOT / "data/interim/pilot/multiseason_40_games.jsonl"),
            pbp_manifest_path=(PROJECT_ROOT / "configs/pbp_40_game_pilot.yaml"),
            canonical_output_root=(PROJECT_ROOT / "data/interim/pbp"),
            per_game_audit_root=(PROJECT_ROOT / "storage/audits/local"),
            output_path=(PROJECT_ROOT / "data/interim/team_game/multiseason_40_team_game.jsonl"),
            audit_path=(PROJECT_ROOT / "storage/audits/local/multiseason_40_team_game.json"),
        )
    except (
        FileNotFoundError,
        TeamGameAggregateError,
    ) as exc:
        print(f"Team-game aggregate build failed: {exc}")
        return 1

    print(json.dumps(result_as_dict(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
