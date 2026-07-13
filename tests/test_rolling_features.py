import hashlib
import json
from pathlib import Path

import pytest

from nhl_ml.rolling_features import (
    RollingFeatureError,
    build_pregame_rolling_features,
)


def team_row(
    *,
    game_id: str,
    season_id: str,
    start: str,
    team_id: str,
    opponent_team_id: str,
    venue_side: str,
    score: int,
    goals: int,
    sog: int,
    attempts: int,
    blocked_attempts: int,
    faceoff_wins: int,
) -> dict:
    return {
        "schema_version": "1.1",
        "batch_id": "fixture_batch",
        "source_selection_id": "fixture_selection",
        "source_selection_sha256": "fixture",
        "game_id": game_id,
        "season_id": season_id,
        "game_type": "regular_season",
        "scheduled_start_utc": start,
        "expected_outcome": "regulation",
        "team_id": team_id,
        "opponent_team_id": opponent_team_id,
        "venue_side": venue_side,
        "official_final_score": score,
        "official_shots_on_goal": sog,
        "pbp_goals_non_shootout": goals,
        "pbp_shots_on_goal": sog,
        "shot_attempt_events": attempts,
        "blocked_shot_attempts": blocked_attempts,
        "penalties": 2,
        "penalty_minutes": 4,
        "faceoff_wins": faceoff_wins,
        "hits": 10,
        "giveaways": 5,
        "takeaways": 4,
    }


def prepare_fixture(
    tmp_path: Path,
) -> tuple[Path, Path]:
    rows = [
        team_row(
            game_id="g1",
            season_id="20242025",
            start="2024-10-01T18:00:00+00:00",
            team_id="1",
            opponent_team_id="2",
            venue_side="home",
            score=3,
            goals=3,
            sog=30,
            attempts=55,
            blocked_attempts=15,
            faceoff_wins=28,
        ),
        team_row(
            game_id="g1",
            season_id="20242025",
            start="2024-10-01T18:00:00+00:00",
            team_id="2",
            opponent_team_id="1",
            venue_side="away",
            score=1,
            goals=1,
            sog=20,
            attempts=40,
            blocked_attempts=10,
            faceoff_wins=22,
        ),
        team_row(
            game_id="g2",
            season_id="20242025",
            start="2024-10-02T17:00:00+00:00",
            team_id="1",
            opponent_team_id="3",
            venue_side="away",
            score=2,
            goals=2,
            sog=25,
            attempts=48,
            blocked_attempts=12,
            faceoff_wins=24,
        ),
        team_row(
            game_id="g2",
            season_id="20242025",
            start="2024-10-02T17:00:00+00:00",
            team_id="3",
            opponent_team_id="1",
            venue_side="home",
            score=4,
            goals=4,
            sog=35,
            attempts=60,
            blocked_attempts=16,
            faceoff_wins=26,
        ),
        team_row(
            game_id="g3",
            season_id="20242025",
            start="2024-10-02T22:00:00+00:00",
            team_id="1",
            opponent_team_id="4",
            venue_side="home",
            score=5,
            goals=5,
            sog=36,
            attempts=64,
            blocked_attempts=18,
            faceoff_wins=30,
        ),
        team_row(
            game_id="g3",
            season_id="20242025",
            start="2024-10-02T22:00:00+00:00",
            team_id="4",
            opponent_team_id="1",
            venue_side="away",
            score=2,
            goals=2,
            sog=27,
            attempts=45,
            blocked_attempts=11,
            faceoff_wins=20,
        ),
        team_row(
            game_id="g4",
            season_id="20242025",
            start="2024-10-03T20:00:00+00:00",
            team_id="1",
            opponent_team_id="5",
            venue_side="away",
            score=0,
            goals=0,
            sog=22,
            attempts=43,
            blocked_attempts=9,
            faceoff_wins=21,
        ),
        team_row(
            game_id="g4",
            season_id="20242025",
            start="2024-10-03T20:00:00+00:00",
            team_id="5",
            opponent_team_id="1",
            venue_side="home",
            score=1,
            goals=1,
            sog=29,
            attempts=50,
            blocked_attempts=13,
            faceoff_wins=29,
        ),
        team_row(
            game_id="g5",
            season_id="20252026",
            start="2025-10-01T20:00:00+00:00",
            team_id="1",
            opponent_team_id="6",
            venue_side="home",
            score=2,
            goals=2,
            sog=31,
            attempts=52,
            blocked_attempts=14,
            faceoff_wins=25,
        ),
        team_row(
            game_id="g5",
            season_id="20252026",
            start="2025-10-01T20:00:00+00:00",
            team_id="6",
            opponent_team_id="1",
            venue_side="away",
            score=3,
            goals=3,
            sog=33,
            attempts=56,
            blocked_attempts=15,
            faceoff_wins=27,
        ),
    ]

    source_path = tmp_path / "team_game.jsonl"
    source_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()

    audit_path = tmp_path / "team_game_audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "batch_id": "fixture_batch",
                "game_count": 5,
                "row_count": 10,
                "output_sha256": source_sha256,
            }
        ),
        encoding="utf-8",
    )

    return source_path, audit_path


def build_fixture(tmp_path: Path):
    source_path, audit_path = prepare_fixture(tmp_path)

    return build_pregame_rolling_features(
        team_game_path=source_path,
        team_game_audit_path=audit_path,
        output_path=tmp_path / "features.jsonl",
        audit_path=tmp_path / "features_audit.json",
    )


def read_rows(result) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(result.output_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_same_utc_date_history_is_excluded(
    tmp_path: Path,
) -> None:
    result = build_fixture(tmp_path)
    rows = read_rows(result)

    team_one_g3 = next(row for row in rows if row["game_id"] == "g3" and row["team_id"] == "1")

    assert team_one_g3["history_game_ids_season_to_date"] == ["g1"]
    assert team_one_g3["same_utc_date_prior_candidates_excluded"] == 1
    assert team_one_g3["season_to_date_games"] == 1


def test_rolling_features_use_prior_games_only(
    tmp_path: Path,
) -> None:
    result = build_fixture(tmp_path)
    rows = read_rows(result)

    team_one_g4 = next(row for row in rows if row["game_id"] == "g4" and row["team_id"] == "1")

    assert team_one_g4["history_game_ids_season_to_date"] == ["g1", "g2", "g3"]
    assert team_one_g4["season_to_date_games"] == 3
    assert team_one_g4["last_3_games"] == 3
    assert team_one_g4["last_5_games"] == 3

    assert team_one_g4["season_to_date_win_rate"] == (2 / 3)
    assert team_one_g4["season_to_date_goals_for_per_game"] == 10 / 3


def test_history_resets_at_season_boundary(
    tmp_path: Path,
) -> None:
    result = build_fixture(tmp_path)
    rows = read_rows(result)

    next_season = next(row for row in rows if row["game_id"] == "g5" and row["team_id"] == "1")

    assert next_season["season_to_date_games"] == 0
    assert next_season["history_game_ids_season_to_date"] == []
    assert next_season["previous_game_id"] is None


def test_build_is_deterministic(
    tmp_path: Path,
) -> None:
    first = build_fixture(tmp_path)
    first_bytes = Path(first.output_path).read_bytes()

    second = build_fixture(tmp_path)

    assert second.already_present is True
    assert second.output_sha256 == first.output_sha256
    assert Path(second.output_path).read_bytes() == first_bytes


def test_source_hash_mismatch_is_rejected(
    tmp_path: Path,
) -> None:
    source_path, audit_path = prepare_fixture(tmp_path)

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["output_sha256"] = "0" * 64
    audit_path.write_text(
        json.dumps(audit),
        encoding="utf-8",
    )

    with pytest.raises(
        RollingFeatureError,
        match="source hash",
    ):
        build_pregame_rolling_features(
            team_game_path=source_path,
            team_game_audit_path=audit_path,
            output_path=tmp_path / "features.jsonl",
            audit_path=tmp_path / "features_audit.json",
        )
