import json
from datetime import date
from pathlib import Path

import pytest

from nhl_ml.sources.nhl_web import (
    NHLRawImportError,
    import_manual_score_file,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = PROJECT_ROOT / "tests/fixtures/nhl_score_minimal.json"


def test_manual_score_import_preserves_raw_bytes(tmp_path: Path) -> None:
    raw_root = tmp_path / "storage/raw"
    manifest_path = tmp_path / "storage/manifests/local/imports.jsonl"

    result = import_manual_score_file(
        input_path=FIXTURE_PATH,
        requested_date=date(2025, 1, 11),
        raw_root=raw_root,
        manifest_path=manifest_path,
    )

    destination = raw_root / result.raw_relative_path

    assert destination.read_bytes() == FIXTURE_PATH.read_bytes()
    assert result.game_count == 1
    assert result.byte_size == FIXTURE_PATH.stat().st_size
    assert result.already_present is False
    assert result.manifest_recorded is True


def test_reimport_is_idempotent(tmp_path: Path) -> None:
    raw_root = tmp_path / "storage/raw"
    manifest_path = tmp_path / "storage/manifests/local/imports.jsonl"

    first = import_manual_score_file(
        input_path=FIXTURE_PATH,
        requested_date=date(2025, 1, 11),
        raw_root=raw_root,
        manifest_path=manifest_path,
    )
    second = import_manual_score_file(
        input_path=FIXTURE_PATH,
        requested_date=date(2025, 1, 11),
        raw_root=raw_root,
        manifest_path=manifest_path,
    )

    manifest_lines = [
        line for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]

    assert first.already_present is False
    assert second.already_present is True
    assert first.sha256 == second.sha256
    assert len(manifest_lines) == 1


def test_manifest_contains_required_audit_fields(tmp_path: Path) -> None:
    manifest_path = tmp_path / "storage/manifests/local/imports.jsonl"

    result = import_manual_score_file(
        input_path=FIXTURE_PATH,
        requested_date=date(2025, 1, 11),
        raw_root=tmp_path / "storage/raw",
        manifest_path=manifest_path,
    )

    record = json.loads(manifest_path.read_text(encoding="utf-8").strip())

    assert record["source_id"] == "nhl_web"
    assert record["import_mode"] == "manual_file"
    assert record["requested_date"] == "2025-01-11"
    assert record["sha256"] == result.sha256
    assert record["game_count"] == 1
    assert "games" in record["top_level_keys"]


def test_wrong_requested_date_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(NHLRawImportError, match="does not match"):
        import_manual_score_file(
            input_path=FIXTURE_PATH,
            requested_date=date(2025, 1, 12),
            raw_root=tmp_path / "storage/raw",
            manifest_path=tmp_path / "imports.jsonl",
        )


def test_payload_without_games_is_rejected(tmp_path: Path) -> None:
    invalid_file = tmp_path / "invalid.json"
    invalid_file.write_text(
        '{"currentDate": "2025-01-11"}',
        encoding="utf-8",
    )

    with pytest.raises(NHLRawImportError, match="'games' list"):
        import_manual_score_file(
            input_path=invalid_file,
            requested_date=date(2025, 1, 11),
            raw_root=tmp_path / "storage/raw",
            manifest_path=tmp_path / "imports.jsonl",
        )


def test_non_json_file_is_rejected(tmp_path: Path) -> None:
    invalid_file = tmp_path / "response.json"
    invalid_file.write_text("<html>Access denied</html>", encoding="utf-8")

    with pytest.raises(NHLRawImportError, match="valid UTF-8 JSON"):
        import_manual_score_file(
            input_path=invalid_file,
            requested_date=date(2025, 1, 11),
            raw_root=tmp_path / "storage/raw",
            manifest_path=tmp_path / "imports.jsonl",
        )
