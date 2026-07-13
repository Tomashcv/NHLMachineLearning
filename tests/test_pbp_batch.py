import json
from pathlib import Path

import pytest

from nhl_ml.pbp_batch import (
    PbpBatchConfig,
    PbpBatchError,
    PbpBatchGame,
    run_pbp_batch,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_FIXTURE = PROJECT_ROOT / "tests/fixtures/nhl_pbp_reconciliation.json"


def make_payload(
    *,
    game_id: str,
    outcome: str,
) -> dict:
    payload = json.loads(BASE_FIXTURE.read_text(encoding="utf-8"))
    payload["id"] = int(game_id)

    if outcome == "overtime_only":
        payload["plays"].insert(
            -2,
            {
                "eventId": 9001,
                "sortOrder": 85,
                "typeDescKey": "period-start",
                "periodDescriptor": {
                    "number": 4,
                    "periodType": "OT",
                },
                "timeInPeriod": "00:00",
                "timeRemaining": "05:00",
            },
        )

    if outcome == "shootout":
        payload["plays"].insert(
            -2,
            {
                "eventId": 9002,
                "sortOrder": 85,
                "typeDescKey": "period-start",
                "periodDescriptor": {
                    "number": 5,
                    "periodType": "SO",
                },
                "timeInPeriod": "00:00",
                "timeRemaining": "00:00",
            },
        )

    return payload


def prepare_batch(
    tmp_path: Path,
) -> tuple[PbpBatchConfig, Path]:
    downloads_dir = tmp_path / "downloads"
    downloads_dir.mkdir(exist_ok=True)

    game_specs = [
        ("2024021001", "regulation"),
        ("2024021002", "regulation"),
        ("2024021003", "regulation"),
        ("2024021004", "overtime_only"),
        ("2024021005", "shootout"),
    ]

    games = []

    for game_id, outcome in game_specs:
        filename = f"nhl_pbp_{game_id}.json"
        path = downloads_dir / filename
        path.write_text(
            json.dumps(
                make_payload(
                    game_id=game_id,
                    outcome=outcome,
                )
            ),
            encoding="utf-8",
        )

        games.append(
            PbpBatchGame(
                game_id=game_id,
                expected_outcome=outcome,
                source_filename=filename,
            )
        )

    config = PbpBatchConfig(
        batch_id="test_batch",
        expected_game_count=5,
        games=tuple(games),
    )
    config.validate()

    return config, downloads_dir


def run_fixture_batch(tmp_path: Path):
    config, downloads_dir = prepare_batch(tmp_path)

    return run_pbp_batch(
        config=config,
        downloads_dir=downloads_dir,
        raw_root=tmp_path / "storage/raw",
        manifest_path=tmp_path / "storage/manifests/imports.jsonl",
        canonical_output_root=tmp_path / "data/interim/pbp",
        per_game_audit_root=tmp_path / "storage/audits",
        aggregate_audit_path=tmp_path / "storage/audits/batch.json",
    )


def test_processes_complete_five_game_batch(
    tmp_path: Path,
) -> None:
    result = run_fixture_batch(tmp_path)

    assert result.processed_game_count == 5
    assert result.regulation_count == 3
    assert result.overtime_only_count == 1
    assert result.shootout_count == 1
    assert result.all_sog_reconciled is True
    assert result.all_applicable_scores_reconciled is True
    assert result.all_outcomes_matched is True
    assert result.all_core_events_have_team is True
    assert result.status == "complete"


def test_batch_audit_has_all_green_gates(
    tmp_path: Path,
) -> None:
    result = run_fixture_batch(tmp_path)
    audit = json.loads(Path(result.audit_path).read_text(encoding="utf-8"))

    assert audit["processed_game_count"] == 5
    assert all(audit["gates"].values())
    assert len(audit["games"]) == 5


def test_batch_is_deterministic(tmp_path: Path) -> None:
    first = run_fixture_batch(tmp_path)
    first_audit = Path(first.audit_path).read_bytes()

    second = run_fixture_batch(tmp_path)

    assert second.batch_sha256 == first.batch_sha256
    assert Path(second.audit_path).read_bytes() == first_audit


def test_missing_download_is_rejected(tmp_path: Path) -> None:
    config, downloads_dir = prepare_batch(tmp_path)
    missing = downloads_dir / config.games[0].source_filename
    missing.unlink()

    with pytest.raises(FileNotFoundError, match="Missing configured"):
        run_pbp_batch(
            config=config,
            downloads_dir=downloads_dir,
            raw_root=tmp_path / "storage/raw",
            manifest_path=tmp_path / "manifest.jsonl",
            canonical_output_root=tmp_path / "canonical",
            per_game_audit_root=tmp_path / "audits",
            aggregate_audit_path=tmp_path / "batch.json",
        )


def test_wrong_expected_outcome_fails_gate(
    tmp_path: Path,
) -> None:
    config, downloads_dir = prepare_batch(tmp_path)

    games = list(config.games)
    games[0] = PbpBatchGame(
        game_id=games[0].game_id,
        expected_outcome="shootout",
        source_filename=games[0].source_filename,
    )

    wrong_config = PbpBatchConfig(
        batch_id="wrong_outcome",
        expected_game_count=5,
        games=tuple(games),
    )

    with pytest.raises(PbpBatchError, match="failed gates"):
        run_pbp_batch(
            config=wrong_config,
            downloads_dir=downloads_dir,
            raw_root=tmp_path / "storage/raw",
            manifest_path=tmp_path / "manifest.jsonl",
            canonical_output_root=tmp_path / "canonical",
            per_game_audit_root=tmp_path / "audits",
            aggregate_audit_path=tmp_path / "batch.json",
        )
