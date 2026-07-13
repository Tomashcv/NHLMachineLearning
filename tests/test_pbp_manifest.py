import hashlib
import json
from pathlib import Path

import pytest
import yaml

from nhl_ml.pbp_manifest import (
    PbpManifestError,
    build_pbp_manifest,
)


def make_record(
    *,
    game_id: str,
    season_id: str,
    game_type: str,
    outcome: str,
    start: str,
) -> dict:
    return {
        "game": {
            "game_id": game_id,
            "season_id": season_id,
            "game_type": game_type,
            "scheduled_start_utc": start,
            "status": "final",
            "went_to_overtime": outcome in {"overtime_only", "shootout"},
            "went_to_shootout": outcome == "shootout",
        },
        "pilot_selection": {
            "selection_id": "test_selection",
        },
    }


def prepare_selection(
    tmp_path: Path,
) -> tuple[Path, Path]:
    records = [
        make_record(
            game_id="1",
            season_id="20202021",
            game_type="regular_season",
            outcome="regulation",
            start="2021-01-01T12:00:00+00:00",
        ),
        make_record(
            game_id="2",
            season_id="20212022",
            game_type="regular_season",
            outcome="overtime_only",
            start="2022-01-01T12:00:00+00:00",
        ),
        make_record(
            game_id="3",
            season_id="20222023",
            game_type="playoff",
            outcome="shootout",
            start="2023-05-01T12:00:00+00:00",
        ),
    ]

    pilot_path = tmp_path / "pilot.jsonl"
    pilot_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )

    digest = hashlib.sha256(pilot_path.read_bytes()).hexdigest()

    audit_path = tmp_path / "selection.json"
    audit_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "selection_id": "test_selection",
                "selection_sha256": digest,
                "selected_game_ids": ["1", "2", "3"],
            }
        ),
        encoding="utf-8",
    )

    return pilot_path, audit_path


def test_builds_deterministic_pbp_manifest(
    tmp_path: Path,
) -> None:
    pilot_path, audit_path = prepare_selection(tmp_path)
    output_path = tmp_path / "manifest.yaml"

    first = build_pbp_manifest(
        pilot_path=pilot_path,
        selection_audit_path=audit_path,
        output_path=output_path,
        batch_id="test_batch",
        expected_game_count=3,
    )
    first_bytes = output_path.read_bytes()

    second = build_pbp_manifest(
        pilot_path=pilot_path,
        selection_audit_path=audit_path,
        output_path=output_path,
        batch_id="test_batch",
        expected_game_count=3,
    )

    payload = yaml.safe_load(output_path.read_text(encoding="utf-8"))

    assert first.game_count == 3
    assert first.regulation_count == 1
    assert first.overtime_only_count == 1
    assert first.shootout_count == 1
    assert first.playoff_count == 1
    assert second.already_present is True
    assert second.manifest_sha256 == first.manifest_sha256
    assert output_path.read_bytes() == first_bytes
    assert payload["games"][2]["expected_outcome"] == "shootout"


def test_rejects_selection_hash_mismatch(
    tmp_path: Path,
) -> None:
    pilot_path, audit_path = prepare_selection(tmp_path)

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["selection_sha256"] = "incorrect"
    audit_path.write_text(
        json.dumps(audit),
        encoding="utf-8",
    )

    with pytest.raises(
        PbpManifestError,
        match="hash does not match",
    ):
        build_pbp_manifest(
            pilot_path=pilot_path,
            selection_audit_path=audit_path,
            output_path=tmp_path / "manifest.yaml",
            batch_id="test_batch",
            expected_game_count=3,
        )
