import hashlib
import json
from pathlib import Path

import pytest

from nhl_ml.model_ready_panel import (
    MODEL_FEATURE_COLUMNS,
    MODEL_FEATURE_FIELDS,
    ModelReadyPanelError,
    build_model_ready_panel,
)


def write_jsonl(
    path: Path,
    records: list[dict],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        "".join(
            json.dumps(
                record,
                sort_keys=True,
            )
            + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def team_row(
    *,
    game_id: str,
    season_id: str,
    split_role: str,
    start: str,
    team_id: str,
    opponent_team_id: str,
    venue_side: str,
    score: int,
) -> dict:
    return {
        "schema_version": "1.2",
        "game_id": game_id,
        "season_id": season_id,
        "split_role": split_role,
        "scheduled_start_utc": start,
        "team_id": team_id,
        "opponent_team_id": (opponent_team_id),
        "venue_side": venue_side,
        "official_final_score": score,
        "expected_outcome": "regulation",
    }


def pregame_row(
    *,
    game_id: str,
    season_id: str,
    start: str,
    team_id: str,
    opponent_team_id: str,
    venue_side: str,
    goals_for: float | None,
    history_games: int,
) -> dict:
    row = {
        "schema_version": "1.0",
        "game_id": game_id,
        "season_id": season_id,
        "scheduled_start_utc": start,
        "team_id": team_id,
        "opponent_team_id": (opponent_team_id),
        "venue_side": venue_side,
        "shots_on_goal_source": ("official_boxscore"),
        "goals_source": ("canonical_pbp_non_shootout"),
    }

    for field in MODEL_FEATURE_FIELDS:
        row[field] = None

    row["season_to_date_games"] = history_games
    row["last_3_games"] = min(
        history_games,
        3,
    )
    row["last_5_games"] = min(
        history_games,
        5,
    )
    row["season_to_date_goals_for_per_game"] = goals_for
    row["days_since_previous_game_start"] = 2.0 if history_games else None

    return row


def prepare_fixture(
    tmp_path: Path,
) -> dict[str, Path]:
    team_rows = [
        team_row(
            game_id="2021020001",
            season_id="20212022",
            split_role="development",
            start=("2021-10-12T20:00:00+00:00"),
            team_id="1",
            opponent_team_id="2",
            venue_side="away",
            score=2,
        ),
        team_row(
            game_id="2021020001",
            season_id="20212022",
            split_role="development",
            start=("2021-10-12T20:00:00+00:00"),
            team_id="2",
            opponent_team_id="1",
            venue_side="home",
            score=4,
        ),
        team_row(
            game_id="2023020001",
            season_id="20232024",
            split_role="validation",
            start=("2023-10-12T20:00:00+00:00"),
            team_id="3",
            opponent_team_id="4",
            venue_side="away",
            score=3,
        ),
        team_row(
            game_id="2023020001",
            season_id="20232024",
            split_role="validation",
            start=("2023-10-12T20:00:00+00:00"),
            team_id="4",
            opponent_team_id="3",
            venue_side="home",
            score=1,
        ),
    ]

    pregame_rows = [
        pregame_row(
            game_id="2021020001",
            season_id="20212022",
            start=("2021-10-12T20:00:00+00:00"),
            team_id="1",
            opponent_team_id="2",
            venue_side="away",
            goals_for=2.0,
            history_games=2,
        ),
        pregame_row(
            game_id="2021020001",
            season_id="20212022",
            start=("2021-10-12T20:00:00+00:00"),
            team_id="2",
            opponent_team_id="1",
            venue_side="home",
            goals_for=3.5,
            history_games=2,
        ),
        pregame_row(
            game_id="2023020001",
            season_id="20232024",
            start=("2023-10-12T20:00:00+00:00"),
            team_id="3",
            opponent_team_id="4",
            venue_side="away",
            goals_for=None,
            history_games=0,
        ),
        pregame_row(
            game_id="2023020001",
            season_id="20232024",
            start=("2023-10-12T20:00:00+00:00"),
            team_id="4",
            opponent_team_id="3",
            venue_side="home",
            goals_for=None,
            history_games=0,
        ),
    ]

    team_path = tmp_path / "team_game.jsonl"
    pregame_path = tmp_path / "pregame.jsonl"

    write_jsonl(team_path, team_rows)
    write_jsonl(pregame_path, pregame_rows)

    team_sha256 = hashlib.sha256(team_path.read_bytes()).hexdigest()
    pregame_sha256 = hashlib.sha256(pregame_path.read_bytes()).hexdigest()

    team_audit_path = tmp_path / "team_game_audit.json"
    team_audit_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "batch_id": "fixture_batch",
                "game_count": 2,
                "row_count": 4,
                "output_sha256": (team_sha256),
            }
        ),
        encoding="utf-8",
    )

    pregame_audit_path = tmp_path / "pregame_audit.json"
    pregame_audit_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "source_batch_id": ("fixture_batch"),
                "source_team_game_sha256": (team_sha256),
                "game_count": 2,
                "row_count": 4,
                "output_sha256": (pregame_sha256),
            }
        ),
        encoding="utf-8",
    )

    return {
        "team_path": team_path,
        "team_audit_path": (team_audit_path),
        "pregame_path": pregame_path,
        "pregame_audit_path": (pregame_audit_path),
        "output_path": (tmp_path / "model_ready.jsonl"),
        "output_audit_path": (tmp_path / "model_ready_audit.json"),
    }


def build_fixture(tmp_path: Path):
    paths = prepare_fixture(tmp_path)

    result = build_model_ready_panel(
        team_game_path=paths["team_path"],
        team_game_audit_path=(paths["team_audit_path"]),
        pregame_path=paths["pregame_path"],
        pregame_audit_path=(paths["pregame_audit_path"]),
        output_path=paths["output_path"],
        audit_path=(paths["output_audit_path"]),
    )

    return result, paths


def test_builds_one_row_per_game_with_separate_targets(
    tmp_path: Path,
) -> None:
    result, paths = build_fixture(tmp_path)

    assert result.status == "complete"
    assert result.game_count == 2
    assert result.row_count == 2
    assert result.base_feature_count == 58
    assert result.feature_column_count == 174

    rows = [
        json.loads(line)
        for line in paths["output_path"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    audit = json.loads(paths["output_audit_path"].read_text(encoding="utf-8"))

    first = rows[0]

    assert first["game_id"] == "2021020001"
    assert first["home_team_id"] == "2"
    assert first["away_team_id"] == "1"
    assert first["target_home_win"] == 1

    assert first["home_season_to_date_goals_for_per_game"] == 3.5
    assert first["away_season_to_date_goals_for_per_game"] == 2.0
    assert first["home_minus_away_season_to_date_goals_for_per_game"] == 1.5

    second = rows[1]

    assert second["target_home_win"] == 0
    assert second["home_minus_away_season_to_date_goals_for_per_game"] is None

    assert not any(column.startswith("target_") for column in audit["feature_columns"])
    assert set(audit["target_columns"]) == {
        "target_home_win",
        "target_home_final_score",
        "target_away_final_score",
        "target_outcome_type",
    }
    assert all(audit["gates"].values())


def test_model_ready_panel_is_deterministic(
    tmp_path: Path,
) -> None:
    first, paths = build_fixture(tmp_path)
    first_bytes = paths["output_path"].read_bytes()

    second, _ = build_fixture(tmp_path)

    assert second.already_present is True
    assert second.output_sha256 == first.output_sha256
    assert paths["output_path"].read_bytes() == first_bytes


def test_rejects_pregame_source_hash_drift(
    tmp_path: Path,
) -> None:
    paths = prepare_fixture(tmp_path)

    paths["pregame_path"].write_text(
        paths["pregame_path"].read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ModelReadyPanelError,
        match=("Pregame feature panel source hash does not match audit"),
    ):
        build_model_ready_panel(
            team_game_path=(paths["team_path"]),
            team_game_audit_path=(paths["team_audit_path"]),
            pregame_path=(paths["pregame_path"]),
            pregame_audit_path=(paths["pregame_audit_path"]),
            output_path=(paths["output_path"]),
            audit_path=(paths["output_audit_path"]),
        )


def test_feature_schema_contains_no_targets() -> None:
    assert len(MODEL_FEATURE_FIELDS) == 58
    assert len(MODEL_FEATURE_COLUMNS) == 174
    assert not any(column.startswith("target_") for column in MODEL_FEATURE_COLUMNS)
