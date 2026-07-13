import hashlib
import json
from pathlib import Path

import pytest
import yaml

from nhl_ml.team_game_aggregates import (
    TeamGameAggregateError,
    build_team_game_aggregates,
)


def event(
    *,
    game_id: str,
    event_id: str,
    event_type: str,
    team_id: str | None,
    period_type: str = "REG",
    shot_type: str | None = None,
    x_coord: int | None = None,
    y_coord: int | None = None,
    penalty_minutes: int | None = None,
    empty_net_candidate: bool = False,
) -> dict:
    return {
        "event": {
            "game_id": game_id,
            "event_id": event_id,
            "event_type": event_type,
            "period_type": period_type,
            "event_owner_team_id": team_id,
            "shot_type": shot_type,
            "x_coord": x_coord,
            "y_coord": y_coord,
            "penalty_duration_minutes": penalty_minutes,
            "empty_net_candidate": empty_net_candidate,
        }
    }


def write_canonical_game(
    *,
    root: Path,
    audit_root: Path,
    game_id: str,
    records: list[dict],
    official_sog: dict[str, int],
) -> None:
    relative_path = Path(f"nhl_web/{game_id}/events_fixture.jsonl")
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )

    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    audit_root.mkdir(parents=True, exist_ok=True)
    (audit_root / f"pbp_canonical_{game_id}.json").write_text(
        json.dumps(
            {
                "game_id": game_id,
                "event_count": len(records),
                "output_relative_path": (relative_path.as_posix()),
                "output_sha256": digest,
                "official_shots_on_goal": official_sog,
            }
        ),
        encoding="utf-8",
    )


def prepare_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path]:
    pilot_records = [
        {
            "game": {
                "game_id": "1001",
                "season_id": "20242025",
                "game_type": "regular_season",
                "scheduled_start_utc": ("2025-01-01T12:00:00+00:00"),
                "home_team_id": "2",
                "away_team_id": "1",
                "home_score": 1,
                "away_score": 2,
                "status": "final",
                "went_to_overtime": False,
                "went_to_shootout": False,
            }
        },
        {
            "game": {
                "game_id": "1002",
                "season_id": "20242025",
                "game_type": "regular_season",
                "scheduled_start_utc": ("2025-01-02T12:00:00+00:00"),
                "home_team_id": "4",
                "away_team_id": "3",
                "home_score": 3,
                "away_score": 2,
                "status": "final",
                "went_to_overtime": True,
                "went_to_shootout": True,
            }
        },
    ]

    pilot_path = tmp_path / "pilot.jsonl"
    pilot_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in pilot_records),
        encoding="utf-8",
    )

    pilot_digest = hashlib.sha256(pilot_path.read_bytes()).hexdigest()

    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "batch": {
                    "batch_id": "test_team_game",
                    "expected_game_count": 2,
                    "source_selection_id": "test_selection",
                    "source_selection_sha256": pilot_digest,
                },
                "games": [
                    {
                        "game_id": "1001",
                        "expected_outcome": "regulation",
                    },
                    {
                        "game_id": "1002",
                        "expected_outcome": "shootout",
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    canonical_root = tmp_path / "canonical"
    audit_root = tmp_path / "audits"

    game_1001 = [
        event(
            game_id="1001",
            event_id="1",
            event_type="shot-on-goal",
            team_id="1",
            shot_type="wrist",
            x_coord=70,
            y_coord=5,
        ),
        event(
            game_id="1001",
            event_id="2",
            event_type="goal",
            team_id="1",
            shot_type="snap",
            x_coord=80,
            y_coord=1,
        ),
        event(
            game_id="1001",
            event_id="3",
            event_type="shot-on-goal",
            team_id="1",
            shot_type="wrist",
            x_coord=65,
            y_coord=-4,
        ),
        event(
            game_id="1001",
            event_id="4",
            event_type="goal",
            team_id="1",
            shot_type=None,
            x_coord=82,
            y_coord=0,
        ),
        event(
            game_id="1001",
            event_id="5",
            event_type="shot-on-goal",
            team_id="2",
            shot_type="slap",
            x_coord=-75,
            y_coord=8,
        ),
        event(
            game_id="1001",
            event_id="6",
            event_type="goal",
            team_id="2",
            shot_type=None,
            x_coord=-80,
            y_coord=0,
        ),
        event(
            game_id="1001",
            event_id="7",
            event_type="missed-shot",
            team_id="1",
            x_coord=60,
            y_coord=20,
        ),
        event(
            game_id="1001",
            event_id="8",
            event_type="blocked-shot",
            team_id="2",
            x_coord=-55,
            y_coord=-10,
        ),
        event(
            game_id="1001",
            event_id="9",
            event_type="penalty",
            team_id="1",
            penalty_minutes=2,
        ),
        event(
            game_id="1001",
            event_id="10",
            event_type="faceoff",
            team_id="2",
        ),
        event(
            game_id="1001",
            event_id="11",
            event_type="hit",
            team_id="1",
        ),
        event(
            game_id="1001",
            event_id="12",
            event_type="giveaway",
            team_id="2",
        ),
        event(
            game_id="1001",
            event_id="13",
            event_type="takeaway",
            team_id="1",
        ),
    ]

    write_canonical_game(
        root=canonical_root,
        audit_root=audit_root,
        game_id="1001",
        records=game_1001,
        official_sog={"1": 3, "2": 1},
    )

    game_1002 = [
        event(
            game_id="1002",
            event_id="1",
            event_type="goal",
            team_id="3",
            shot_type="wrist",
            x_coord=80,
            y_coord=0,
        ),
        event(
            game_id="1002",
            event_id="2",
            event_type="goal",
            team_id="3",
            shot_type="snap",
            x_coord=82,
            y_coord=2,
        ),
        event(
            game_id="1002",
            event_id="3",
            event_type="goal",
            team_id="4",
            shot_type="wrist",
            x_coord=-80,
            y_coord=0,
        ),
        event(
            game_id="1002",
            event_id="4",
            event_type="goal",
            team_id="4",
            shot_type="snap",
            x_coord=-82,
            y_coord=-2,
        ),
        event(
            game_id="1002",
            event_id="5",
            event_type="shootout-complete",
            team_id=None,
            period_type="SO",
        ),
    ]

    write_canonical_game(
        root=canonical_root,
        audit_root=audit_root,
        game_id="1002",
        records=game_1002,
        official_sog={"3": 2, "4": 2},
    )

    return (
        pilot_path,
        manifest_path,
        canonical_root,
        audit_root,
    )


def build_fixture(tmp_path: Path):
    (
        pilot_path,
        manifest_path,
        canonical_root,
        audit_root,
    ) = prepare_fixture(tmp_path)

    return build_team_game_aggregates(
        pilot_path=pilot_path,
        pbp_manifest_path=manifest_path,
        canonical_output_root=canonical_root,
        per_game_audit_root=audit_root,
        output_path=tmp_path / "team_game.jsonl",
        audit_path=tmp_path / "team_game_audit.json",
    )


def test_builds_two_rows_per_game(
    tmp_path: Path,
) -> None:
    result = build_fixture(tmp_path)

    assert result.game_count == 2
    assert result.row_count == 4
    assert result.all_sog_reconciled is True
    assert result.all_applicable_scores_reconciled is True
    assert result.status == "complete"


def test_team_metrics_are_aggregated(
    tmp_path: Path,
) -> None:
    result = build_fixture(tmp_path)

    rows = [
        json.loads(line)
        for line in Path(result.output_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    away = next(row for row in rows if row["game_id"] == "1001" and row["team_id"] == "1")

    assert away["venue_side"] == "away"
    assert away["pbp_goals_non_shootout"] == 2
    assert away["pbp_shots_on_goal"] == 3
    assert away["shot_on_goal_events"] == 2
    assert away["goals_with_shot_type"] == 1
    assert away["goals_without_shot_type"] == 1
    assert away["shot_attempt_events"] == 5
    assert away["missed_shots"] == 1
    assert away["penalties"] == 1
    assert away["penalty_minutes"] == 2
    assert away["hits"] == 1
    assert away["takeaways"] == 1


def test_shootout_score_reconciliation_is_not_required(
    tmp_path: Path,
) -> None:
    result = build_fixture(tmp_path)

    rows = [
        json.loads(line)
        for line in Path(result.output_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    shootout_rows = [row for row in rows if row["game_id"] == "1002"]

    assert len(shootout_rows) == 2
    assert all(row["score_reconciliation_applicable"] is False for row in shootout_rows)
    assert all(row["score_reconciliation_passed"] is True for row in shootout_rows)


def test_build_is_deterministic(tmp_path: Path) -> None:
    first = build_fixture(tmp_path)
    first_bytes = Path(first.output_path).read_bytes()

    second = build_fixture(tmp_path)

    assert second.already_present is True
    assert second.output_sha256 == first.output_sha256
    assert Path(second.output_path).read_bytes() == first_bytes


def test_unknown_event_owner_team_is_rejected(
    tmp_path: Path,
) -> None:
    (
        pilot_path,
        manifest_path,
        canonical_root,
        audit_root,
    ) = prepare_fixture(tmp_path)

    canonical_path = canonical_root / "nhl_web/1001/events_fixture.jsonl"
    records = [
        json.loads(line)
        for line in canonical_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    records.append(
        event(
            game_id="1001",
            event_id="999",
            event_type="shot-on-goal",
            team_id="999",
            shot_type="wrist",
        )
    )

    canonical_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )

    digest = hashlib.sha256(canonical_path.read_bytes()).hexdigest()
    audit_path = audit_root / "pbp_canonical_1001.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["event_count"] = len(records)
    audit["output_sha256"] = digest
    audit_path.write_text(
        json.dumps(audit),
        encoding="utf-8",
    )

    with pytest.raises(
        TeamGameAggregateError,
        match="gates failed",
    ):
        build_team_game_aggregates(
            pilot_path=pilot_path,
            pbp_manifest_path=manifest_path,
            canonical_output_root=canonical_root,
            per_game_audit_root=audit_root,
            output_path=tmp_path / "output.jsonl",
            audit_path=tmp_path / "audit.json",
        )


def test_blocked_shot_is_an_attempt_by_owner_team(
    tmp_path: Path,
) -> None:
    result = build_fixture(tmp_path)

    rows = [
        json.loads(line)
        for line in Path(result.output_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    away = next(row for row in rows if row["game_id"] == "1001" and row["team_id"] == "1")
    home = next(row for row in rows if row["game_id"] == "1001" and row["team_id"] == "2")

    assert "blocked_shots" not in away
    assert "blocked_shots" not in home

    assert away["blocked_shot_attempts"] == 0
    assert home["blocked_shot_attempts"] == 1


def test_shot_attempt_identity_includes_goals_without_shot_type(
    tmp_path: Path,
) -> None:
    result = build_fixture(tmp_path)

    rows = [
        json.loads(line)
        for line in Path(result.output_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    for row in rows:
        assert row["shot_attempt_events"] == (
            row["pbp_shots_on_goal"]
            + row["goals_without_shot_type"]
            + row["missed_shots"]
            + row["blocked_shot_attempts"]
        )

        assert row["unblocked_shot_attempt_events"] == (
            row["pbp_shots_on_goal"] + row["goals_without_shot_type"] + row["missed_shots"]
        )

    audit = json.loads(Path(result.audit_path).read_text(encoding="utf-8"))

    assert audit["gates"]["passes_shot_attempt_identity"] is True
