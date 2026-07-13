import json
from pathlib import Path

import pytest

from nhl_ml.pbp_raw import PbpRawError, process_manual_pbp_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = PROJECT_ROOT / "tests/fixtures/nhl_pbp_minimal.json"


def process_fixture(tmp_path: Path):
    return process_manual_pbp_file(
        input_path=FIXTURE,
        expected_game_id="2024020669",
        raw_root=tmp_path / "storage/raw",
        manifest_path=tmp_path / "storage/manifests/imports.jsonl",
        audit_path=tmp_path / "storage/audits/pbp.json",
    )


def test_imports_and_audits_pbp_file(tmp_path: Path) -> None:
    result = process_fixture(tmp_path)

    raw_file = tmp_path / "storage/raw" / result.raw_relative_path
    audit = json.loads(Path(result.audit_path).read_text(encoding="utf-8"))

    assert raw_file.read_bytes() == FIXTURE.read_bytes()
    assert result.game_id == "2024020669"
    assert result.play_count == 4
    assert audit["play_count"] == 4
    assert audit["period_type_counts"] == {"OT": 1, "REG": 3}
    assert audit["plays_with_coordinates"] == 3
    assert audit["event_type_counts"]["goal"] == 2


def test_reimport_is_idempotent(tmp_path: Path) -> None:
    first = process_fixture(tmp_path)
    second = process_fixture(tmp_path)

    manifest = tmp_path / "storage/manifests/imports.jsonl"
    lines = [line for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert first.already_present is False
    assert second.already_present is True
    assert first.raw_sha256 == second.raw_sha256
    assert len(lines) == 1


def test_wrong_game_id_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(PbpRawError, match="does not match"):
        process_manual_pbp_file(
            input_path=FIXTURE,
            expected_game_id="9999999999",
            raw_root=tmp_path / "raw",
            manifest_path=tmp_path / "manifest.jsonl",
            audit_path=tmp_path / "audit.json",
        )


def test_missing_plays_is_rejected(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.json"
    invalid.write_text(
        json.dumps(
            {
                "id": 2024020669,
                "startTimeUTC": "2025-01-11T18:00:00Z",
                "homeTeam": {"id": 13},
                "awayTeam": {"id": 6},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(PbpRawError, match="plays list"):
        process_manual_pbp_file(
            input_path=invalid,
            expected_game_id="2024020669",
            raw_root=tmp_path / "raw",
            manifest_path=tmp_path / "manifest.jsonl",
            audit_path=tmp_path / "audit.json",
        )


def test_sort_order_audit_is_complete(tmp_path: Path) -> None:
    result = process_fixture(tmp_path)
    audit = json.loads(Path(result.audit_path).read_text(encoding="utf-8"))

    assert audit["missing_event_id_count"] == 0
    assert audit["duplicate_event_ids"] == []
    assert audit["missing_sort_order_count"] == 0
    assert audit["sort_order_complete"] is True
    assert audit["sort_order_nondecreasing"] is True
