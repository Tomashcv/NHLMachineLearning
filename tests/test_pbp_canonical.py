import hashlib
import json
from pathlib import Path

import pytest

from nhl_ml.pbp_canonical import (
    PbpCanonicalizationError,
    canonicalize_pbp_file,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = PROJECT_ROOT / "tests/fixtures/nhl_pbp_reconciliation.json"


def prepare_raw_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, Path]:
    raw_root = tmp_path / "storage/raw"
    relative_path = Path("nhl_web/manual/play_by_play/2024020999/pbp_fixture.json")
    raw_file = raw_root / relative_path
    raw_file.parent.mkdir(parents=True)
    raw_file.write_bytes(FIXTURE.read_bytes())

    digest = hashlib.sha256(raw_file.read_bytes()).hexdigest()

    manifest_path = tmp_path / "storage/manifests/local/imports.jsonl"
    manifest_path.parent.mkdir(parents=True)

    manifest_path.write_text(
        json.dumps(
            {
                "endpoint_family": "play_by_play",
                "game_id": "2024020999",
                "raw_relative_path": relative_path.as_posix(),
                "sha256": digest,
                "imported_at_utc": "2025-02-02T12:00:00+00:00",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    return raw_root, raw_file, manifest_path


def canonicalize_fixture(tmp_path: Path):
    raw_root, raw_file, manifest_path = prepare_raw_fixture(tmp_path)

    return canonicalize_pbp_file(
        raw_file=raw_file,
        raw_root=raw_root,
        import_manifest_path=manifest_path,
        output_root=tmp_path / "data/interim/pbp",
        audit_path=tmp_path / "audits/pbp.json",
    )


def test_reconciles_score_and_shots_on_goal(
    tmp_path: Path,
) -> None:
    result = canonicalize_fixture(tmp_path)
    audit = json.loads(Path(result.audit_path).read_text(encoding="utf-8"))

    assert result.event_count == 10
    assert result.goal_count == 3
    assert result.shot_on_goal_event_count == 2
    assert result.score_reconciliation_passed is True
    assert result.sog_reconciliation_passed is True

    assert audit["official_scores"] == {
        "13": 1,
        "6": 2,
    }
    assert audit["pbp_goal_counts_excluding_shootout"] == {
        "13": 1,
        "6": 2,
    }
    assert audit["official_shots_on_goal"] == {
        "13": 2,
        "6": 3,
    }
    assert audit["pbp_shots_on_goal_including_goals"] == {
        "13": 2,
        "6": 3,
    }


def test_detects_empty_net_goal_candidate(
    tmp_path: Path,
) -> None:
    result = canonicalize_fixture(tmp_path)
    output_path = tmp_path / "data/interim/pbp" / result.output_relative_path

    records = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    empty_net_events = [
        record["event"] for record in records if record["event"]["empty_net_candidate"]
    ]

    assert result.empty_net_goal_candidate_count == 1
    assert len(empty_net_events) == 1
    assert empty_net_events[0]["event_id"] == "7"


def test_canonicalization_is_deterministic(
    tmp_path: Path,
) -> None:
    raw_root, raw_file, manifest_path = prepare_raw_fixture(tmp_path)
    output_root = tmp_path / "data/interim/pbp"
    audit_path = tmp_path / "audits/pbp.json"

    first = canonicalize_pbp_file(
        raw_file=raw_file,
        raw_root=raw_root,
        import_manifest_path=manifest_path,
        output_root=output_root,
        audit_path=audit_path,
    )
    first_bytes = (output_root / first.output_relative_path).read_bytes()

    second = canonicalize_pbp_file(
        raw_file=raw_file,
        raw_root=raw_root,
        import_manifest_path=manifest_path,
        output_root=output_root,
        audit_path=audit_path,
    )

    assert second.already_present is True
    assert second.output_sha256 == first.output_sha256
    assert (output_root / second.output_relative_path).read_bytes() == first_bytes


def test_duplicate_event_id_is_rejected(
    tmp_path: Path,
) -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    payload["plays"][1]["eventId"] = payload["plays"][0]["eventId"]

    modified_fixture = tmp_path / "duplicate.json"
    modified_fixture.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    raw_root = tmp_path / "storage/raw"
    relative_path = Path("nhl_web/manual/play_by_play/2024020999/pbp_duplicate.json")
    raw_file = raw_root / relative_path
    raw_file.parent.mkdir(parents=True)
    raw_file.write_bytes(modified_fixture.read_bytes())

    digest = hashlib.sha256(raw_file.read_bytes()).hexdigest()
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "endpoint_family": "play_by_play",
                "game_id": "2024020999",
                "raw_relative_path": relative_path.as_posix(),
                "sha256": digest,
                "imported_at_utc": "2025-02-02T12:00:00+00:00",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        PbpCanonicalizationError,
        match="Duplicate event ID",
    ):
        canonicalize_pbp_file(
            raw_file=raw_file,
            raw_root=raw_root,
            import_manifest_path=manifest_path,
            output_root=tmp_path / "output",
            audit_path=tmp_path / "audit.json",
        )
