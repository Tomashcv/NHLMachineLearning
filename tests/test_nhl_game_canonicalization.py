import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from nhl_ml.canonical.nhl_games import (
    NHLCanonicalizationError,
    canonicalize_score_file,
    parse_score_payload,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTCOMES_FIXTURE = PROJECT_ROOT / "tests/fixtures/nhl_score_outcomes.json"


def load_outcomes_payload() -> dict:
    return json.loads(OUTCOMES_FIXTURE.read_text(encoding="utf-8"))


def observed_time() -> datetime:
    return datetime(2025, 1, 12, 12, tzinfo=UTC)


def test_parses_regulation_overtime_and_shootout() -> None:
    games = parse_score_payload(
        payload=load_outcomes_payload(),
        source_observed_at_utc=observed_time(),
        ingested_at_utc=observed_time(),
    )

    assert len(games) == 3

    regulation, overtime, shootout = games

    assert regulation.went_to_overtime is False
    assert regulation.went_to_shootout is False

    assert overtime.went_to_overtime is True
    assert overtime.went_to_shootout is False

    assert shootout.went_to_overtime is True
    assert shootout.went_to_shootout is True


def test_game_ids_and_team_ids_are_strings() -> None:
    games = parse_score_payload(
        payload=load_outcomes_payload(),
        source_observed_at_utc=observed_time(),
        ingested_at_utc=observed_time(),
    )

    assert games[0].game_id == "2024020681"
    assert games[0].home_team_id == "2"
    assert games[0].away_team_id == "1"
    assert games[0].scheduled_start_utc.tzinfo is UTC


def test_unknown_game_state_is_rejected() -> None:
    payload = load_outcomes_payload()
    payload["games"][0]["gameState"] = "UNKNOWN"

    with pytest.raises(
        NHLCanonicalizationError,
        match="Unsupported NHL game state",
    ):
        parse_score_payload(
            payload=payload,
            source_observed_at_utc=observed_time(),
            ingested_at_utc=observed_time(),
        )


def test_duplicate_game_id_is_rejected() -> None:
    payload = load_outcomes_payload()
    payload["games"][1]["id"] = payload["games"][0]["id"]

    with pytest.raises(
        NHLCanonicalizationError,
        match="Duplicate game ID",
    ):
        parse_score_payload(
            payload=payload,
            source_observed_at_utc=observed_time(),
            ingested_at_utc=observed_time(),
        )


def test_final_game_without_outcome_is_rejected() -> None:
    payload = load_outcomes_payload()
    del payload["games"][0]["gameOutcome"]

    with pytest.raises(
        NHLCanonicalizationError,
        match="gameOutcome",
    ):
        parse_score_payload(
            payload=payload,
            source_observed_at_utc=observed_time(),
            ingested_at_utc=observed_time(),
        )


def test_canonicalization_is_deterministic(tmp_path: Path) -> None:
    raw_root = tmp_path / "storage/raw"
    raw_relative_path = Path("nhl_web/manual/2025-01-11/score_fixture.json")
    raw_file = raw_root / raw_relative_path
    raw_file.parent.mkdir(parents=True)
    raw_file.write_bytes(OUTCOMES_FIXTURE.read_bytes())

    raw_digest = hashlib.sha256(raw_file.read_bytes()).hexdigest()

    manifest_path = tmp_path / "storage/manifests/local/imports.jsonl"
    manifest_path.parent.mkdir(parents=True)

    manifest_record = {
        "requested_date": "2025-01-11",
        "raw_relative_path": raw_relative_path.as_posix(),
        "sha256": raw_digest,
        "imported_at_utc": "2025-01-12T12:00:00+00:00",
    }
    manifest_path.write_text(
        json.dumps(manifest_record) + "\n",
        encoding="utf-8",
    )

    output_root = tmp_path / "data/interim"

    first = canonicalize_score_file(
        raw_file=raw_file,
        raw_root=raw_root,
        import_manifest_path=manifest_path,
        output_root=output_root,
    )
    second = canonicalize_score_file(
        raw_file=raw_file,
        raw_root=raw_root,
        import_manifest_path=manifest_path,
        output_root=output_root,
    )

    output_file = output_root / first.output_relative_path
    lines = output_file.read_text(encoding="utf-8").splitlines()

    assert first.already_present is False
    assert second.already_present is True
    assert first.output_sha256 == second.output_sha256
    assert first.game_count == 3
    assert first.regulation_count == 1
    assert first.overtime_only_count == 1
    assert first.shootout_count == 1
    assert len(lines) == 3
