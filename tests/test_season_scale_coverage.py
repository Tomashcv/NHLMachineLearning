import json
from pathlib import Path

import yaml

from nhl_ml.season_scale_coverage import (
    audit_season_scale_coverage,
)
from nhl_ml.season_scale_manifest import (
    build_season_scale_manifest,
)


def prepare_manifest(
    tmp_path: Path,
) -> tuple[Path, Path]:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0",
                "corpus": {
                    "corpus_id": "fixture_corpus",
                    "game_type_code": "02",
                    "expected_game_count": 2,
                    "playoffs_included": False,
                    "split_policy": ("frozen_by_season"),
                },
                "seasons": [
                    {
                        "season_id": "20212022",
                        "start_year": 2021,
                        "first_sequence": 1,
                        "last_sequence": 2,
                        "expected_game_count": 2,
                        "split_role": "development",
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    manifest_path = tmp_path / "manifest.jsonl"
    manifest_audit_path = tmp_path / "manifest_audit.json"

    build_season_scale_manifest(
        config_path=config_path,
        manifest_path=manifest_path,
        audit_path=manifest_audit_path,
    )

    return manifest_path, manifest_audit_path


def valid_payload(
    *,
    game_id: int,
) -> dict:
    return {
        "id": game_id,
        "season": 20212022,
        "gameType": 2,
        "gameState": "OFF",
        "startTimeUTC": ("2021-10-01T23:00:00Z"),
        "awayTeam": {"id": 1},
        "homeTeam": {"id": 2},
        "plays": [{"eventId": 1}],
    }


def test_partial_coverage_is_reported(
    tmp_path: Path,
) -> None:
    (
        manifest_path,
        manifest_audit_path,
    ) = prepare_manifest(tmp_path)

    downloads = tmp_path / "downloads"
    downloads.mkdir()

    (downloads / "nhl_pbp_2021020001.json").write_text(
        json.dumps(valid_payload(game_id=2021020001)),
        encoding="utf-8",
    )

    result = audit_season_scale_coverage(
        manifest_path=manifest_path,
        manifest_audit_path=(manifest_audit_path),
        downloads_dirs=[downloads],
        audit_path=tmp_path / "coverage.json",
        missing_ids_path=(tmp_path / "missing.txt"),
    )

    assert result.status == "partial"
    assert result.expected_game_count == 2
    assert result.valid_game_count == 1
    assert result.missing_game_count == 1
    assert result.invalid_file_count == 0


def test_wrong_payload_id_is_invalid(
    tmp_path: Path,
) -> None:
    (
        manifest_path,
        manifest_audit_path,
    ) = prepare_manifest(tmp_path)

    downloads = tmp_path / "downloads"
    downloads.mkdir()

    (downloads / "nhl_pbp_2021020001.json").write_text(
        json.dumps(valid_payload(game_id=2021029999)),
        encoding="utf-8",
    )

    result = audit_season_scale_coverage(
        manifest_path=manifest_path,
        manifest_audit_path=(manifest_audit_path),
        downloads_dirs=[downloads],
        audit_path=tmp_path / "coverage.json",
        missing_ids_path=(tmp_path / "missing.txt"),
    )

    assert result.status == "invalid"
    assert result.valid_game_count == 0
    assert result.invalid_file_count == 1
    assert result.missing_game_count == 1
