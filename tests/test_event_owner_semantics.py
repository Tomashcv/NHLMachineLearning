import hashlib
import json
from pathlib import Path

import pytest
import yaml

from nhl_ml.event_owner_semantics import (
    EventOwnerSemanticError,
    audit_event_owner_semantics,
)


def canonical_event(
    *,
    event_id: str,
    event_type: str,
    owner_team_id: str,
    player_ids: dict[str, str],
) -> dict:
    return {
        "event": {
            "game_id": "1001",
            "event_id": event_id,
            "event_type": event_type,
            "event_owner_team_id": owner_team_id,
            "period_number": 1,
            "period_type": "REG",
            "time_in_period": "01:00",
            "player_ids": player_ids,
        }
    }


def prepare_fixture(
    tmp_path: Path,
    *,
    mixed_blocked_semantics: bool = False,
) -> tuple[Path, Path, Path, Path]:
    raw_root = tmp_path / "raw"
    canonical_root = tmp_path / "canonical"
    audit_root = tmp_path / "audits"

    raw_relative_path = Path("nhl_web/manual/play_by_play/1001/pbp_fixture.json")
    raw_path = raw_root / raw_relative_path
    raw_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_payload = {
        "id": 1001,
        "homeTeam": {"id": 2},
        "awayTeam": {"id": 1},
        "rosterSpots": [
            {"playerId": 11, "teamId": 1},
            {"playerId": 12, "teamId": 1},
            {"playerId": 21, "teamId": 2},
            {"playerId": 22, "teamId": 2},
        ],
    }

    raw_path.write_text(
        json.dumps(raw_payload),
        encoding="utf-8",
    )
    raw_digest = hashlib.sha256(raw_path.read_bytes()).hexdigest()

    records = [
        canonical_event(
            event_id="1",
            event_type="blocked-shot",
            owner_team_id="1",
            player_ids={
                "shooting_player_id": "11",
                "blocking_player_id": "21",
            },
        ),
        canonical_event(
            event_id="2",
            event_type="faceoff",
            owner_team_id="1",
            player_ids={
                "winning_player_id": "11",
                "losing_player_id": "21",
            },
        ),
        canonical_event(
            event_id="3",
            event_type="hit",
            owner_team_id="1",
            player_ids={
                "hitting_player_id": "12",
                "hittee_player_id": "22",
            },
        ),
        canonical_event(
            event_id="4",
            event_type="giveaway",
            owner_team_id="1",
            player_ids={"player_id": "11"},
        ),
        canonical_event(
            event_id="5",
            event_type="takeaway",
            owner_team_id="2",
            player_ids={"player_id": "21"},
        ),
    ]

    if mixed_blocked_semantics:
        records.append(
            canonical_event(
                event_id="6",
                event_type="blocked-shot",
                owner_team_id="2",
                player_ids={
                    "shooting_player_id": "12",
                    "blocking_player_id": "22",
                },
            )
        )

    canonical_relative_path = Path("nhl_web/1001/events_fixture.jsonl")
    canonical_path = canonical_root / canonical_relative_path
    canonical_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    canonical_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    canonical_digest = hashlib.sha256(canonical_path.read_bytes()).hexdigest()

    audit_root.mkdir(
        parents=True,
        exist_ok=True,
    )
    (audit_root / "pbp_canonical_1001.json").write_text(
        json.dumps(
            {
                "game_id": "1001",
                "raw_relative_path": (raw_relative_path.as_posix()),
                "raw_sha256": raw_digest,
                "output_relative_path": (canonical_relative_path.as_posix()),
                "output_sha256": canonical_digest,
            }
        ),
        encoding="utf-8",
    )

    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "batch": {
                    "batch_id": "semantic_test",
                    "expected_game_count": 1,
                },
                "games": [{"game_id": "1001"}],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    return (
        manifest_path,
        raw_root,
        canonical_root,
        audit_root,
    )


def run_fixture(
    tmp_path: Path,
    *,
    mixed_blocked_semantics: bool = False,
):
    (
        manifest_path,
        raw_root,
        canonical_root,
        audit_root,
    ) = prepare_fixture(
        tmp_path,
        mixed_blocked_semantics=(mixed_blocked_semantics),
    )

    return audit_event_owner_semantics(
        pbp_manifest_path=manifest_path,
        raw_root=raw_root,
        canonical_output_root=canonical_root,
        per_game_audit_root=audit_root,
        output_path=tmp_path / "semantics.jsonl",
        audit_path=tmp_path / "semantic_audit.json",
    )


def test_infers_event_owner_roles(
    tmp_path: Path,
) -> None:
    result = run_fixture(tmp_path)

    audit = json.loads(Path(result.audit_path).read_text(encoding="utf-8"))

    assert result.status == "complete"
    assert result.validation_row_count == 5

    assert audit["event_types"]["blocked-shot"]["inferred_owner_role"] == "shooting_player_id"

    assert audit["event_types"]["faceoff"]["inferred_owner_role"] == "winning_player_id"

    assert audit["event_types"]["hit"]["inferred_owner_role"] == "hitting_player_id"

    assert audit["event_types"]["giveaway"]["inferred_owner_role"] == "player_id"

    assert audit["event_types"]["takeaway"]["inferred_owner_role"] == "player_id"


def test_semantic_audit_is_deterministic(
    tmp_path: Path,
) -> None:
    first = run_fixture(tmp_path)
    first_bytes = Path(first.output_path).read_bytes()

    second = run_fixture(tmp_path)

    assert second.already_present is True
    assert second.output_sha256 == first.output_sha256
    assert Path(second.output_path).read_bytes() == first_bytes


def test_mixed_semantics_fail_gate(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        EventOwnerSemanticError,
        match="semantic gates failed",
    ):
        run_fixture(
            tmp_path,
            mixed_blocked_semantics=True,
        )


def test_unmapped_player_fails_gate(
    tmp_path: Path,
) -> None:
    (
        manifest_path,
        raw_root,
        canonical_root,
        audit_root,
    ) = prepare_fixture(tmp_path)

    canonical_path = canonical_root / "nhl_web/1001/events_fixture.jsonl"
    records = [
        json.loads(line)
        for line in canonical_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    records[0]["event"]["player_ids"]["shooting_player_id"] = "999"

    canonical_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )

    digest = hashlib.sha256(canonical_path.read_bytes()).hexdigest()

    audit_path = audit_root / "pbp_canonical_1001.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["output_sha256"] = digest
    audit_path.write_text(
        json.dumps(audit),
        encoding="utf-8",
    )

    with pytest.raises(
        EventOwnerSemanticError,
        match="semantic gates failed",
    ):
        audit_event_owner_semantics(
            pbp_manifest_path=manifest_path,
            raw_root=raw_root,
            canonical_output_root=canonical_root,
            per_game_audit_root=audit_root,
            output_path=tmp_path / "output.jsonl",
            audit_path=tmp_path / "audit.json",
        )


def test_same_team_secondary_role_remains_unambiguous(
    tmp_path: Path,
) -> None:
    (
        manifest_path,
        raw_root,
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
        canonical_event(
            event_id="6",
            event_type="blocked-shot",
            owner_team_id="1",
            player_ids={
                "shooting_player_id": "11",
                "blocking_player_id": "12",
            },
        )
    )

    canonical_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )

    digest = hashlib.sha256(canonical_path.read_bytes()).hexdigest()

    canonical_audit_path = audit_root / "pbp_canonical_1001.json"
    canonical_audit = json.loads(canonical_audit_path.read_text(encoding="utf-8"))
    canonical_audit["output_sha256"] = digest
    canonical_audit_path.write_text(
        json.dumps(canonical_audit),
        encoding="utf-8",
    )

    result = audit_event_owner_semantics(
        pbp_manifest_path=manifest_path,
        raw_root=raw_root,
        canonical_output_root=canonical_root,
        per_game_audit_root=audit_root,
        output_path=tmp_path / "output.jsonl",
        audit_path=tmp_path / "audit.json",
    )

    audit = json.loads(Path(result.audit_path).read_text(encoding="utf-8"))
    blocked = audit["event_types"]["blocked-shot"]

    assert result.status == "complete"
    assert blocked["event_count"] == 2
    assert blocked["role_match_counts"] == {
        "blocking_player_id": 1,
        "shooting_player_id": 2,
    }
    assert blocked["same_team_multi_role_event_count"] == 1
    assert blocked["inferred_owner_role"] == "shooting_player_id"
