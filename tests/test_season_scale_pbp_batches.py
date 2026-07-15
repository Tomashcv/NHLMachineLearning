import hashlib
import json
from pathlib import Path

import pytest

from nhl_ml.season_scale_pbp_batches import (
    SeasonScalePbpBatchError,
    build_season_scale_pbp_batch_configs,
    verify_season_pbp_batch_config,
)


def inventory_row(
    *,
    game_id: str,
    season_id: str,
    split_role: str,
    start: str,
    outcome: str,
) -> dict:
    return {
        "schema_version": "1.0",
        "corpus_id": "fixture_corpus",
        "game_id": game_id,
        "season_id": season_id,
        "split_role": split_role,
        "sequence_number": int(game_id[-4:]),
        "game_type": "regular_season",
        "game_state": "OFF",
        "scheduled_start_utc": start,
        "away_team_id": "1",
        "home_team_id": "2",
        "away_score": 2,
        "home_score": 3,
        "away_shots_on_goal": 28,
        "home_shots_on_goal": 31,
        "provider_last_period_type": "REG",
        "expected_outcome": outcome,
        "play_count": 1,
        "roster_spot_count": 2,
        "source_filename": f"nhl_pbp_{game_id}.json",
        "source_path": f"/tmp/nhl_pbp_{game_id}.json",
        "source_sha256": game_id.zfill(64),
        "source_byte_size": 100,
    }


def prepare_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, Path]:
    rows = [
        inventory_row(
            game_id="2021020001",
            season_id="20212022",
            split_role="development",
            start="2021-10-01T20:00:00+00:00",
            outcome="regulation",
        ),
        inventory_row(
            game_id="2021020002",
            season_id="20212022",
            split_role="development",
            start="2021-10-02T20:00:00+00:00",
            outcome="overtime_only",
        ),
        inventory_row(
            game_id="2022020001",
            season_id="20222023",
            split_role="validation",
            start="2022-10-01T20:00:00+00:00",
            outcome="shootout",
        ),
        inventory_row(
            game_id="2022020002",
            season_id="20222023",
            split_role="validation",
            start="2022-10-02T20:00:00+00:00",
            outcome="regulation",
        ),
    ]

    inventory_path = tmp_path / "inventory.jsonl"
    inventory_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    inventory_sha256 = hashlib.sha256(inventory_path.read_bytes()).hexdigest()

    inventory_audit_path = tmp_path / "inventory_audit.json"
    inventory_audit_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "corpus_id": "fixture_corpus",
                "game_count": 4,
                "season_counts": {
                    "20212022": 2,
                    "20222023": 2,
                },
                "output_sha256": inventory_sha256,
            }
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "configs"

    return inventory_path, inventory_audit_path, output_dir


def build_fixture(tmp_path: Path):
    inventory_path, audit_path, output_dir = prepare_fixture(tmp_path)

    result = build_season_scale_pbp_batch_configs(
        inventory_path=inventory_path,
        inventory_audit_path=audit_path,
        output_dir=output_dir,
    )

    return result, inventory_path, audit_path, output_dir


def test_builds_one_config_per_season(
    tmp_path: Path,
) -> None:
    result, inventory_path, audit_path, output_dir = build_fixture(tmp_path)

    assert result.status == "complete"
    assert result.season_count == 2
    assert result.game_count == 4

    verification = verify_season_pbp_batch_config(
        season_id="20212022",
        config_path=output_dir / "pbp_season_20212022.yaml",
        inventory_path=inventory_path,
        inventory_audit_path=audit_path,
    )

    assert verification["status"] == "verified"
    assert verification["game_count"] == 2
    assert verification["unique_game_count"] == 2
    assert verification["outcome_counts"] == {
        "overtime_only": 1,
        "regulation": 1,
    }


def test_batch_configs_are_deterministic(
    tmp_path: Path,
) -> None:
    first, _, _, _ = build_fixture(tmp_path)

    second, _, _, _ = build_fixture(tmp_path)

    assert second.already_present_count == 2
    assert second.config_sha256s == first.config_sha256s
    assert second.subset_sha256s == first.subset_sha256s


def test_inventory_drift_is_rejected(
    tmp_path: Path,
) -> None:
    _, inventory_path, audit_path, output_dir = build_fixture(tmp_path)

    rows = [
        json.loads(line)
        for line in inventory_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows[0]["expected_outcome"] = "shootout"

    inventory_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    updated_sha256 = hashlib.sha256(inventory_path.read_bytes()).hexdigest()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["output_sha256"] = updated_sha256
    audit_path.write_text(
        json.dumps(audit),
        encoding="utf-8",
    )

    with pytest.raises(
        SeasonScalePbpBatchError,
        match="does not match verified inventory",
    ):
        verify_season_pbp_batch_config(
            season_id="20212022",
            config_path=(output_dir / "pbp_season_20212022.yaml"),
            inventory_path=inventory_path,
            inventory_audit_path=audit_path,
        )
