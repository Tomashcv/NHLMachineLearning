import json
from pathlib import Path

import pytest

from nhl_ml.pbp_batch import (
    PbpBatchConfig,
    PbpBatchGame,
)
from nhl_ml.pbp_coverage import (
    PbpCoverageError,
    audit_pbp_download_coverage,
)


def make_config() -> PbpBatchConfig:
    config = PbpBatchConfig(
        batch_id="coverage_test",
        expected_game_count=2,
        games=(
            PbpBatchGame(
                game_id="1001",
                expected_outcome="regulation",
                source_filename="nhl_pbp_1001.json",
            ),
            PbpBatchGame(
                game_id="1002",
                expected_outcome="overtime_only",
                source_filename="nhl_pbp_1002.json",
            ),
        ),
    )
    config.validate()
    return config


def write_pbp(
    path: Path,
    *,
    game_id: str,
) -> None:
    path.write_text(
        json.dumps(
            {
                "id": int(game_id),
                "season": 20242025,
                "gameType": 2,
                "gameState": "OFF",
                "plays": [{"eventId": 1}],
            }
        ),
        encoding="utf-8",
    )


def test_complete_coverage_passes(
    tmp_path: Path,
) -> None:
    downloads = tmp_path / "downloads"
    downloads.mkdir()

    write_pbp(
        downloads / "nhl_pbp_1001.json",
        game_id="1001",
    )
    write_pbp(
        downloads / "nhl_pbp_1002.json",
        game_id="1002",
    )

    result = audit_pbp_download_coverage(
        config=make_config(),
        downloads_dir=downloads,
        audit_path=tmp_path / "audit.json",
        require_complete=True,
    )

    assert result.status == "complete"
    assert result.valid_game_count == 2
    assert result.coverage_fraction == 1.0


def test_partial_coverage_reports_missing_game(
    tmp_path: Path,
) -> None:
    downloads = tmp_path / "downloads"
    downloads.mkdir()

    write_pbp(
        downloads / "nhl_pbp_1001.json",
        game_id="1001",
    )

    audit_path = tmp_path / "audit.json"

    result = audit_pbp_download_coverage(
        config=make_config(),
        downloads_dir=downloads,
        audit_path=audit_path,
    )

    audit = json.loads(audit_path.read_text(encoding="utf-8"))

    assert result.status == "partial"
    assert result.valid_game_count == 1
    assert result.missing_game_count == 1
    assert audit["missing_games"][0]["game_id"] == "1002"


def test_wrong_payload_game_id_is_invalid(
    tmp_path: Path,
) -> None:
    downloads = tmp_path / "downloads"
    downloads.mkdir()

    write_pbp(
        downloads / "nhl_pbp_1001.json",
        game_id="9999",
    )
    write_pbp(
        downloads / "nhl_pbp_1002.json",
        game_id="1002",
    )

    with pytest.raises(
        PbpCoverageError,
        match="not complete",
    ):
        audit_pbp_download_coverage(
            config=make_config(),
            downloads_dir=downloads,
            audit_path=tmp_path / "audit.json",
            require_complete=True,
        )
