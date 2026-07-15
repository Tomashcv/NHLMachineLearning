import hashlib
import json
from pathlib import Path

import pytest

from nhl_ml.season_scale_pbp_batches import (
    build_season_scale_pbp_batch_configs,
)
from nhl_ml.season_scale_team_game_aggregates import (
    build_season_scale_team_game_aggregates,
)
from nhl_ml.team_game_aggregates import (
    TeamGameAggregateError,
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


def event(
    *,
    game_id: str,
    event_id: int,
    event_type: str,
    team_id: str,
    shot_type: str | None = None,
) -> dict:
    return {
        "event": {
            "game_id": game_id,
            "event_id": str(event_id),
            "event_type": event_type,
            "period_type": "REG",
            "event_owner_team_id": team_id,
            "shot_type": shot_type,
            "x_coord": 10.0,
            "y_coord": 5.0,
            "empty_net_candidate": False,
            "penalty_duration_minutes": None,
        }
    }


def inventory_row(
    *,
    game_id: str,
    start: str,
    away_score: int,
    home_score: int,
    away_sog: int,
    home_sog: int,
) -> dict:
    return {
        "schema_version": "1.0",
        "corpus_id": "fixture_corpus",
        "game_id": game_id,
        "season_id": "20212022",
        "split_role": "development",
        "sequence_number": int(game_id[-4:]),
        "game_type": "regular_season",
        "game_state": "OFF",
        "scheduled_start_utc": start,
        "away_team_id": "1",
        "home_team_id": "2",
        "away_score": away_score,
        "home_score": home_score,
        "away_shots_on_goal": away_sog,
        "home_shots_on_goal": home_sog,
        "provider_last_period_type": "REG",
        "expected_outcome": "regulation",
        "play_count": 3,
        "roster_spot_count": 2,
        "source_filename": (f"nhl_pbp_{game_id}.json"),
        "source_path": (f"/tmp/nhl_pbp_{game_id}.json"),
        "source_sha256": game_id.zfill(64),
        "source_byte_size": 100,
    }


def prepare_fixture(
    tmp_path: Path,
) -> dict[str, Path]:
    inventory_rows = [
        inventory_row(
            game_id="2021020001",
            start="2021-10-01T20:00:00+00:00",
            away_score=1,
            home_score=0,
            away_sog=2,
            home_sog=1,
        ),
        inventory_row(
            game_id="2021020002",
            start="2021-10-02T20:00:00+00:00",
            away_score=0,
            home_score=1,
            away_sog=1,
            home_sog=1,
        ),
    ]

    inventory_path = tmp_path / "inventory.jsonl"
    write_jsonl(
        inventory_path,
        inventory_rows,
    )

    inventory_sha256 = hashlib.sha256(inventory_path.read_bytes()).hexdigest()

    inventory_audit_path = tmp_path / "inventory_audit.json"
    inventory_audit_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "corpus_id": "fixture_corpus",
                "game_count": 2,
                "season_counts": {
                    "20212022": 2,
                },
                "output_sha256": (inventory_sha256),
            }
        ),
        encoding="utf-8",
    )

    config_dir = tmp_path / "configs"

    build_season_scale_pbp_batch_configs(
        inventory_path=inventory_path,
        inventory_audit_path=(inventory_audit_path),
        output_dir=config_dir,
    )

    canonical_root = tmp_path / "canonical"
    audit_root = tmp_path / "audits"
    audit_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    games = {
        "2021020001": {
            "events": [
                event(
                    game_id="2021020001",
                    event_id=1,
                    event_type="goal",
                    team_id="1",
                    shot_type="wrist",
                ),
                event(
                    game_id="2021020001",
                    event_id=2,
                    event_type="shot-on-goal",
                    team_id="1",
                ),
                event(
                    game_id="2021020001",
                    event_id=3,
                    event_type="shot-on-goal",
                    team_id="2",
                ),
            ],
            "official_sog": {
                "1": 2,
                "2": 1,
            },
            "status": "exact",
            "deltas": {
                "1": 0,
                "2": 0,
            },
            "correction_team": None,
        },
        "2021020002": {
            "events": [
                event(
                    game_id="2021020002",
                    event_id=1,
                    event_type="shot-on-goal",
                    team_id="1",
                ),
                event(
                    game_id="2021020002",
                    event_id=2,
                    event_type="shot-on-goal",
                    team_id="1",
                ),
                event(
                    game_id="2021020002",
                    event_id=3,
                    event_type="goal",
                    team_id="2",
                    shot_type="snap",
                ),
            ],
            "official_sog": {
                "1": 1,
                "2": 1,
            },
            "status": ("provider_boxscore_minus_one_correction"),
            "deltas": {
                "1": 1,
                "2": 0,
            },
            "correction_team": "1",
        },
    }

    for game_id, game in games.items():
        relative_path = Path("fixture") / f"{game_id}.jsonl"
        canonical_path = canonical_root / relative_path
        write_jsonl(
            canonical_path,
            game["events"],
        )
        canonical_sha256 = hashlib.sha256(canonical_path.read_bytes()).hexdigest()

        audit = {
            "game_id": game_id,
            "output_relative_path": (relative_path.as_posix()),
            "output_sha256": (canonical_sha256),
            "event_count": len(game["events"]),
            "official_shots_on_goal": (game["official_sog"]),
            "sog_reconciliation_status": (game["status"]),
            "sog_deltas_by_team": (game["deltas"]),
            "sog_provider_correction_team_id": (game["correction_team"]),
            "sog_reconciliation_passed": True,
        }

        (audit_root / f"pbp_canonical_{game_id}.json").write_text(
            json.dumps(audit),
            encoding="utf-8",
        )

    return {
        "inventory_path": inventory_path,
        "inventory_audit_path": (inventory_audit_path),
        "config_path": (config_dir / "pbp_season_20212022.yaml"),
        "canonical_root": canonical_root,
        "audit_root": audit_root,
        "output_path": (tmp_path / "team_game.jsonl"),
        "output_audit_path": (tmp_path / "team_game_audit.json"),
    }


def build_fixture(tmp_path: Path):
    paths = prepare_fixture(tmp_path)

    result = build_season_scale_team_game_aggregates(
        season_id="20212022",
        config_path=paths["config_path"],
        inventory_path=(paths["inventory_path"]),
        inventory_audit_path=(paths["inventory_audit_path"]),
        canonical_output_root=(paths["canonical_root"]),
        per_game_audit_root=(paths["audit_root"]),
        output_path=paths["output_path"],
        audit_path=(paths["output_audit_path"]),
    )

    return result, paths


def test_builds_two_rows_per_game_and_preserves_sog_policy(
    tmp_path: Path,
) -> None:
    result, paths = build_fixture(tmp_path)

    assert result.status == "complete"
    assert result.game_count == 2
    assert result.row_count == 4
    assert result.all_sog_reconciled is True

    rows = [
        json.loads(line)
        for line in paths["output_path"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    audit = json.loads(paths["output_audit_path"].read_text(encoding="utf-8"))

    corrected = [row for row in rows if row["sog_provider_correction_applied"]]

    assert len(corrected) == 1
    assert corrected[0]["game_id"] == ("2021020002")
    assert corrected[0]["team_id"] == "1"
    assert corrected[0]["official_shots_on_goal"] == 1
    assert corrected[0]["pbp_shots_on_goal"] == 2
    assert corrected[0]["sog_delta_pbp_minus_official"] == 1

    assert audit["sog_reconciliation_status_counts"] == {
        "exact": 1,
        "provider_boxscore_minus_one_correction": 1,
    }
    assert all(audit["gates"].values())


def test_season_scale_team_game_is_deterministic(
    tmp_path: Path,
) -> None:
    first, paths = build_fixture(tmp_path)
    first_bytes = paths["output_path"].read_bytes()

    second, _ = build_fixture(tmp_path)

    assert second.already_present is True
    assert second.output_sha256 == (first.output_sha256)
    assert paths["output_path"].read_bytes() == first_bytes


def test_rejects_canonical_sog_policy_drift(
    tmp_path: Path,
) -> None:
    paths = prepare_fixture(tmp_path)
    audit_path = paths["audit_root"] / "pbp_canonical_2021020002.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["sog_reconciliation_status"] = "exact"
    audit_path.write_text(
        json.dumps(audit),
        encoding="utf-8",
    )

    with pytest.raises(
        TeamGameAggregateError,
        match="Canonical SOG policy mismatch",
    ):
        build_season_scale_team_game_aggregates(
            season_id="20212022",
            config_path=paths["config_path"],
            inventory_path=(paths["inventory_path"]),
            inventory_audit_path=(paths["inventory_audit_path"]),
            canonical_output_root=(paths["canonical_root"]),
            per_game_audit_root=(paths["audit_root"]),
            output_path=paths["output_path"],
            audit_path=(paths["output_audit_path"]),
        )
