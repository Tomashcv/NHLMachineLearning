import hashlib
import json
from pathlib import Path

import pytest

from nhl_ml.season_scale_inventory import (
    SeasonScaleInventoryError,
    build_verified_season_scale_inventory,
)


def payload(
    *,
    game_id: int,
    season_id: int,
    last_period_type: str,
    away_score: int,
    home_score: int,
) -> dict:
    return {
        "id": game_id,
        "season": season_id,
        "gameType": 2,
        "gameState": "OFF",
        "startTimeUTC": (
            "2022-01-01T20:00:00Z" if game_id == 2021020001 else "2022-01-02T20:00:00Z"
        ),
        "awayTeam": {
            "id": 1,
            "score": away_score,
            "sog": 28,
        },
        "homeTeam": {
            "id": 2,
            "score": home_score,
            "sog": 31,
        },
        "gameOutcome": {
            "lastPeriodType": last_period_type,
        },
        "periodDescriptor": {
            "periodType": last_period_type,
        },
        "plays": [{"eventId": 1}],
        "rosterSpots": [
            {"playerId": 11, "teamId": 1},
            {"playerId": 21, "teamId": 2},
        ],
    }


def prepare_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, Path]:
    manifest_rows = [
        {
            "schema_version": "1.0",
            "corpus_id": "fixture_corpus",
            "game_id": "2021020001",
            "season_id": "20212022",
            "split_role": "development",
            "sequence_number": 1,
            "source_filename": ("nhl_pbp_2021020001.json"),
        },
        {
            "schema_version": "1.0",
            "corpus_id": "fixture_corpus",
            "game_id": "2021020002",
            "season_id": "20212022",
            "split_role": "development",
            "sequence_number": 2,
            "source_filename": ("nhl_pbp_2021020002.json"),
        },
    ]

    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )

    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    audit_path = tmp_path / "manifest_audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "corpus_id": "fixture_corpus",
                "game_count": 2,
                "season_counts": {
                    "20212022": 2,
                },
                "split_counts": {
                    "development": 2,
                },
                "manifest_sha256": manifest_sha256,
            }
        ),
        encoding="utf-8",
    )

    downloads_dir = tmp_path / "downloads"
    downloads_dir.mkdir(exist_ok=True)

    (downloads_dir / "nhl_pbp_2021020001.json").write_text(
        json.dumps(
            payload(
                game_id=2021020001,
                season_id=20212022,
                last_period_type="REG",
                away_score=2,
                home_score=4,
            )
        ),
        encoding="utf-8",
    )

    (downloads_dir / "nhl_pbp_2021020002.json").write_text(
        json.dumps(
            payload(
                game_id=2021020002,
                season_id=20212022,
                last_period_type="SO",
                away_score=3,
                home_score=2,
            )
        ),
        encoding="utf-8",
    )

    return manifest_path, audit_path, downloads_dir


def build_fixture(tmp_path: Path):
    (
        manifest_path,
        audit_path,
        downloads_dir,
    ) = prepare_fixture(tmp_path)

    return build_verified_season_scale_inventory(
        manifest_path=manifest_path,
        manifest_audit_path=audit_path,
        downloads_dir=downloads_dir,
        output_path=tmp_path / "inventory.jsonl",
        audit_path=tmp_path / "inventory_audit.json",
    )


def test_builds_verified_inventory(
    tmp_path: Path,
) -> None:
    result = build_fixture(tmp_path)

    assert result.status == "complete"
    assert result.game_count == 2
    assert result.season_count == 1
    assert result.unique_team_count == 2
    assert result.regulation_count == 1
    assert result.shootout_count == 1


def test_inventory_is_deterministic(
    tmp_path: Path,
) -> None:
    first = build_fixture(tmp_path)
    first_bytes = Path(first.output_path).read_bytes()

    second = build_fixture(tmp_path)

    assert second.already_present is True
    assert second.output_sha256 == first.output_sha256
    assert Path(second.output_path).read_bytes() == first_bytes


def test_wrong_payload_season_is_rejected(
    tmp_path: Path,
) -> None:
    (
        manifest_path,
        audit_path,
        downloads_dir,
    ) = prepare_fixture(tmp_path)

    path = downloads_dir / "nhl_pbp_2021020001.json"
    broken = json.loads(path.read_text(encoding="utf-8"))
    broken["season"] = 20222023
    path.write_text(
        json.dumps(broken),
        encoding="utf-8",
    )

    with pytest.raises(
        SeasonScaleInventoryError,
        match="payload season",
    ):
        build_verified_season_scale_inventory(
            manifest_path=manifest_path,
            manifest_audit_path=audit_path,
            downloads_dir=downloads_dir,
            output_path=tmp_path / "inventory.jsonl",
            audit_path=tmp_path / "inventory_audit.json",
        )
