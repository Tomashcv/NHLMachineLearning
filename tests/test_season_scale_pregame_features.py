import hashlib
import json
from pathlib import Path

import pytest

from nhl_ml.rolling_features import (
    RollingFeatureError,
    build_pregame_rolling_features,
)
from nhl_ml.season_scale_pregame_features import (
    build_season_scale_team_game_panel,
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
    start: str,
    team_id: str,
    opponent_team_id: str,
    venue_side: str,
    score: int,
    goals: int,
    official_sog: int,
    pbp_sog: int,
    split_role: str,
) -> dict:
    return {
        "schema_version": "1.2",
        "batch_id": (f"fixture_{season_id}"),
        "corpus_id": "fixture_corpus",
        "source_selection_id": (f"fixture_corpus:{season_id}"),
        "source_selection_sha256": (season_id.zfill(64)),
        "source_inventory_sha256": ("a" * 64),
        "split_role": split_role,
        "game_id": game_id,
        "season_id": season_id,
        "game_type": "regular_season",
        "scheduled_start_utc": start,
        "expected_outcome": "regulation",
        "team_id": team_id,
        "opponent_team_id": (opponent_team_id),
        "venue_side": venue_side,
        "official_final_score": score,
        "official_shots_on_goal": (official_sog),
        "pbp_goals_non_shootout": goals,
        "pbp_shots_on_goal": pbp_sog,
        "shot_attempt_events": (pbp_sog + 10),
        "blocked_shot_attempts": 5,
        "penalties": 2,
        "penalty_minutes": 4,
        "faceoff_wins": 25,
        "hits": 20,
        "giveaways": 8,
        "takeaways": 6,
    }


def prepare_fixture(
    tmp_path: Path,
) -> dict[str, Path]:
    team_game_root = tmp_path / "team_game"
    audit_root = tmp_path / "audits"

    team_game_root.mkdir(
        parents=True,
        exist_ok=True,
    )
    audit_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    season_rows = {
        "20212022": [
            team_row(
                game_id="2021020001",
                season_id="20212022",
                start=("2021-10-01T20:00:00+00:00"),
                team_id="1",
                opponent_team_id="2",
                venue_side="away",
                score=3,
                goals=3,
                official_sog=30,
                pbp_sog=31,
                split_role="development",
            ),
            team_row(
                game_id="2021020001",
                season_id="20212022",
                start=("2021-10-01T20:00:00+00:00"),
                team_id="2",
                opponent_team_id="1",
                venue_side="home",
                score=2,
                goals=2,
                official_sog=25,
                pbp_sog=25,
                split_role="development",
            ),
            team_row(
                game_id="2021020002",
                season_id="20212022",
                start=("2021-10-03T20:00:00+00:00"),
                team_id="1",
                opponent_team_id="2",
                venue_side="home",
                score=1,
                goals=1,
                official_sog=27,
                pbp_sog=27,
                split_role="development",
            ),
            team_row(
                game_id="2021020002",
                season_id="20212022",
                start=("2021-10-03T20:00:00+00:00"),
                team_id="2",
                opponent_team_id="1",
                venue_side="away",
                score=2,
                goals=2,
                official_sog=29,
                pbp_sog=29,
                split_role="development",
            ),
        ],
        "20222023": [
            team_row(
                game_id="2022020001",
                season_id="20222023",
                start=("2022-10-01T20:00:00+00:00"),
                team_id="1",
                opponent_team_id="2",
                venue_side="away",
                score=2,
                goals=2,
                official_sog=28,
                pbp_sog=28,
                split_role="validation",
            ),
            team_row(
                game_id="2022020001",
                season_id="20222023",
                start=("2022-10-01T20:00:00+00:00"),
                team_id="2",
                opponent_team_id="1",
                venue_side="home",
                score=1,
                goals=1,
                official_sog=24,
                pbp_sog=24,
                split_role="validation",
            ),
        ],
    }

    for season_id, rows in season_rows.items():
        source_path = team_game_root / f"team_game_{season_id}.jsonl"
        write_jsonl(source_path, rows)

        source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()

        audit = {
            "status": "complete",
            "season_id": season_id,
            "split_role": rows[0]["split_role"],
            "game_count": len({row["game_id"] for row in rows}),
            "row_count": len(rows),
            "output_sha256": (source_sha256),
        }

        (audit_root / f"team_game_{season_id}.json").write_text(
            json.dumps(audit),
            encoding="utf-8",
        )

    return {
        "team_game_root": team_game_root,
        "audit_root": audit_root,
        "combined_path": (tmp_path / "combined.jsonl"),
        "combined_audit_path": (tmp_path / "combined_audit.json"),
        "feature_path": (tmp_path / "features.jsonl"),
        "feature_audit_path": (tmp_path / "features_audit.json"),
    }


def build_fixture(tmp_path: Path):
    paths = prepare_fixture(tmp_path)

    panel = build_season_scale_team_game_panel(
        season_ids=(
            "20212022",
            "20222023",
        ),
        team_game_root=(paths["team_game_root"]),
        source_audit_root=(paths["audit_root"]),
        output_path=paths["combined_path"],
        audit_path=(paths["combined_audit_path"]),
        expected_team_games_per_season=None,
    )

    features = build_pregame_rolling_features(
        team_game_path=(paths["combined_path"]),
        team_game_audit_path=(paths["combined_audit_path"]),
        output_path=paths["feature_path"],
        audit_path=(paths["feature_audit_path"]),
    )

    return panel, features, paths


def test_combines_seasons_and_resets_history(
    tmp_path: Path,
) -> None:
    panel, features, paths = build_fixture(tmp_path)

    assert panel.status == "complete"
    assert panel.season_count == 2
    assert panel.game_count == 3
    assert panel.row_count == 6

    assert features.status == "complete"
    assert features.row_count == 6

    rows = [
        json.loads(line)
        for line in paths["feature_path"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    lookup = {
        (
            row["game_id"],
            row["team_id"],
        ): row
        for row in rows
    }

    second_game = lookup[("2021020002", "1")]

    assert second_game["season_to_date_shots_on_goal_for_per_game"] == 30.0
    assert second_game["shots_on_goal_source"] == "official_boxscore"

    next_season = lookup[("2022020001", "1")]

    assert next_season["season_to_date_games"] == 0
    assert next_season["history_game_ids_season_to_date"] == []


def test_season_scale_pregame_is_deterministic(
    tmp_path: Path,
) -> None:
    first_panel, first_features, paths = build_fixture(tmp_path)
    first_panel_bytes = paths["combined_path"].read_bytes()
    first_feature_bytes = paths["feature_path"].read_bytes()

    second_panel, second_features, _ = build_fixture(tmp_path)

    assert second_panel.already_present is True
    assert second_panel.output_sha256 == first_panel.output_sha256
    assert second_features.already_present is True
    assert second_features.output_sha256 == first_features.output_sha256
    assert paths["combined_path"].read_bytes() == first_panel_bytes
    assert paths["feature_path"].read_bytes() == first_feature_bytes


def test_rejects_source_hash_drift(
    tmp_path: Path,
) -> None:
    paths = prepare_fixture(tmp_path)

    source_path = paths["team_game_root"] / "team_game_20212022.jsonl"
    source_path.write_text(
        source_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        RollingFeatureError,
        match="source hash mismatch",
    ):
        build_season_scale_team_game_panel(
            season_ids=(
                "20212022",
                "20222023",
            ),
            team_game_root=(paths["team_game_root"]),
            source_audit_root=(paths["audit_root"]),
            output_path=(paths["combined_path"]),
            audit_path=(paths["combined_audit_path"]),
            expected_team_games_per_season=None,
        )
