import hashlib
import json
from pathlib import Path

import pytest

from nhl_ml.experiment_dataset import (
    HOLDOUT_UNLOCK_TOKEN,
    SANITIZED_METADATA_COLUMNS,
    TARGET_COLUMN,
    ExperimentDatasetError,
    SealedHoldoutError,
    build_experiment_dataset_bundle,
    load_experiment_dataset,
)


FEATURE_COLUMNS = (
    "home_form",
    "away_form",
    "home_minus_away_form",
)


def feature_schema_sha256() -> str:
    return hashlib.sha256(
        json.dumps(
            FEATURE_COLUMNS,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def write_jsonl(
    path: Path,
    rows: list[dict],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        "".join(
            json.dumps(
                row,
                sort_keys=True,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def model_ready_row(
    *,
    game_id: str,
    season_id: str,
    split_role: str,
    start: str,
    home_team_id: str,
    away_team_id: str,
    target: int,
) -> dict:
    return {
        "schema_version": "1.0",
        "dataset_id": "fixture_model_ready",
        "game_id": game_id,
        "season_id": season_id,
        "split_role": split_role,
        "scheduled_start_utc": start,
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "home_form": 0.6,
        "away_form": 0.4,
        "home_minus_away_form": 0.2,
        "target_home_win": target,
        "target_home_final_score": 4,
        "target_away_final_score": 2,
        "target_outcome_type": "regulation",
    }


def prepare_fixture(
    tmp_path: Path,
) -> dict[str, Path]:
    rows = [
        model_ready_row(
            game_id="2021020001",
            season_id="20212022",
            split_role="development",
            start="2021-10-12T20:00:00+00:00",
            home_team_id="1",
            away_team_id="2",
            target=1,
        ),
        model_ready_row(
            game_id="2023020001",
            season_id="20232024",
            split_role="validation",
            start="2023-10-12T20:00:00+00:00",
            home_team_id="3",
            away_team_id="4",
            target=0,
        ),
        model_ready_row(
            game_id="2024020001",
            season_id="20242025",
            split_role="sealed_holdout",
            start="2024-10-12T20:00:00+00:00",
            home_team_id="5",
            away_team_id="6",
            target=1,
        ),
    ]

    source_path = tmp_path / "model_ready.jsonl"
    write_jsonl(source_path, rows)

    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()

    source_audit_path = tmp_path / "model_ready_audit.json"
    source_audit_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "dataset_id": ("fixture_model_ready"),
                "game_count": 3,
                "row_count": 3,
                "feature_columns": list(FEATURE_COLUMNS),
                "feature_columns_sha256": (feature_schema_sha256()),
                "target_columns": [
                    "target_home_win",
                    "target_home_final_score",
                    "target_away_final_score",
                    "target_outcome_type",
                ],
                "split_game_counts": {
                    "development": 1,
                    "validation": 1,
                    "sealed_holdout": 1,
                },
                "output_sha256": (source_sha256),
            }
        ),
        encoding="utf-8",
    )

    return {
        "source_path": source_path,
        "source_audit_path": (source_audit_path),
        "output_dir": (tmp_path / "experiment"),
        "bundle_audit_path": (tmp_path / "bundle_audit.json"),
    }


def build_fixture(
    tmp_path: Path,
):
    paths = prepare_fixture(tmp_path)

    result = build_experiment_dataset_bundle(
        source_path=paths["source_path"],
        source_audit_path=(paths["source_audit_path"]),
        output_dir=paths["output_dir"],
        audit_path=(paths["bundle_audit_path"]),
    )

    return result, paths


def test_builds_sanitized_split_bundle_and_default_loader(
    tmp_path: Path,
) -> None:
    result, paths = build_fixture(tmp_path)

    assert result.status == "complete"
    assert result.game_count == 3
    assert result.split_counts == {
        "development": 1,
        "validation": 1,
        "sealed_holdout": 1,
    }

    development_path = paths["output_dir"] / "development.jsonl"
    development_row = json.loads(development_path.read_text(encoding="utf-8").strip())

    expected_columns = {
        *SANITIZED_METADATA_COLUMNS,
        *FEATURE_COLUMNS,
        TARGET_COLUMN,
    }

    assert set(development_row) == (expected_columns)
    assert "target_home_final_score" not in development_row
    assert "target_outcome_type" not in development_row

    holdout_path = paths["output_dir"] / "sealed_holdout.jsonl"
    holdout_path.unlink()

    loaded = load_experiment_dataset(bundle_audit_path=(paths["bundle_audit_path"]))

    assert loaded.row_count == 1
    assert loaded.requested_splits == ("development",)
    assert loaded.split_counts == {"development": 1}
    assert loaded.holdout_unlocked is False


def test_sealed_holdout_requires_two_key_unlock(
    tmp_path: Path,
) -> None:
    _, paths = build_fixture(tmp_path)

    with pytest.raises(
        SealedHoldoutError,
        match="sealed holdout remains locked",
    ):
        load_experiment_dataset(
            bundle_audit_path=(paths["bundle_audit_path"]),
            requested_splits=("sealed_holdout",),
        )

    with pytest.raises(
        SealedHoldoutError,
    ):
        load_experiment_dataset(
            bundle_audit_path=(paths["bundle_audit_path"]),
            requested_splits=("sealed_holdout",),
            allow_sealed_holdout=True,
            holdout_unlock_token="wrong",
        )

    loaded = load_experiment_dataset(
        bundle_audit_path=(paths["bundle_audit_path"]),
        requested_splits=("sealed_holdout",),
        allow_sealed_holdout=True,
        holdout_unlock_token=(HOLDOUT_UNLOCK_TOKEN),
    )

    assert loaded.row_count == 1
    assert loaded.holdout_unlocked is True


def test_loader_rejects_requested_split_hash_drift(
    tmp_path: Path,
) -> None:
    _, paths = build_fixture(tmp_path)

    development_path = paths["output_dir"] / "development.jsonl"
    development_path.write_text(
        development_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ExperimentDatasetError,
        match="Split hash mismatch",
    ):
        load_experiment_dataset(bundle_audit_path=(paths["bundle_audit_path"]))


def test_bundle_build_is_deterministic(
    tmp_path: Path,
) -> None:
    first, paths = build_fixture(tmp_path)

    first_bytes = {
        split_role: (paths["output_dir"] / f"{split_role}.jsonl").read_bytes()
        for split_role in (
            "development",
            "validation",
            "sealed_holdout",
        )
    }

    second = build_experiment_dataset_bundle(
        source_path=paths["source_path"],
        source_audit_path=(paths["source_audit_path"]),
        output_dir=paths["output_dir"],
        audit_path=(paths["bundle_audit_path"]),
    )

    assert first.status == "complete"
    assert second.status == "complete"
    assert second.already_present is True

    for split_role, expected in first_bytes.items():
        assert (paths["output_dir"] / f"{split_role}.jsonl").read_bytes() == expected
