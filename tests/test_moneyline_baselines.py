import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from nhl_ml.experiment_dataset import (
    SANITIZED_METADATA_COLUMNS,
    TARGET_COLUMN,
)
from nhl_ml.moneyline_baselines import (
    PRIMARY_LOGISTIC_FEATURES,
    MoneylineBaselineError,
    probability_metrics,
    run_moneyline_baseline_selection,
)


def feature_schema_sha256(
    columns: tuple[str, ...],
) -> str:
    return hashlib.sha256(
        json.dumps(
            columns,
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


def development_row(
    *,
    game_id: str,
    season_id: str,
    start: str,
    target: int,
    signal: float,
) -> dict:
    row = {
        "schema_version": "1.0",
        "dataset_id": ("fixture_model_ready"),
        "game_id": game_id,
        "season_id": season_id,
        "split_role": "development",
        "scheduled_start_utc": start,
        "home_team_id": f"H{game_id}",
        "away_team_id": f"A{game_id}",
        TARGET_COLUMN: target,
    }

    for index, feature in enumerate(PRIMARY_LOGISTIC_FEATURES):
        row[feature] = signal + index * 0.001

    return row


def prepare_fixture(
    tmp_path: Path,
    *,
    feature_columns: tuple[str, ...] = (PRIMARY_LOGISTIC_FEATURES),
) -> Path:
    rows = []

    for index, target in enumerate(
        (0, 1, 0, 1),
        start=1,
    ):
        rows.append(
            development_row(
                game_id=f"202102000{index}",
                season_id="20212022",
                start=(f"2021-10-{10 + index:02d}T20:00:00+00:00"),
                target=target,
                signal=float(target),
            )
        )

    for index, target in enumerate(
        (1, 0, 1, 0),
        start=1,
    ):
        rows.append(
            development_row(
                game_id=f"202202000{index}",
                season_id="20222023",
                start=(f"2022-10-{10 + index:02d}T20:00:00+00:00"),
                target=target,
                signal=float(target),
            )
        )

    development_path = tmp_path / "development.jsonl"
    write_jsonl(
        development_path,
        rows,
    )

    development_sha256 = hashlib.sha256(development_path.read_bytes()).hexdigest()

    bundle_audit_path = tmp_path / "bundle_audit.json"
    bundle_audit_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "bundle_id": ("fixture_bundle"),
                "status": "complete",
                "source_dataset_id": ("fixture_model_ready"),
                "feature_columns": list(feature_columns),
                "feature_columns_sha256": (feature_schema_sha256(feature_columns)),
                "metadata_columns": list(SANITIZED_METADATA_COLUMNS),
                "target_column": (TARGET_COLUMN),
                "split_files": {
                    "development": {
                        "path": str(development_path),
                        "row_count": 8,
                        "sha256": (development_sha256),
                    },
                    "validation": {
                        "path": str(tmp_path / "missing_validation.jsonl"),
                        "row_count": 0,
                        "sha256": "not-read",
                    },
                    "sealed_holdout": {
                        "path": str(tmp_path / "missing_holdout.jsonl"),
                        "row_count": 0,
                        "sha256": "not-read",
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    return bundle_audit_path


def test_probability_metrics_known_values() -> None:
    metrics = probability_metrics(
        np.asarray([0, 1]),
        np.asarray([0.25, 0.75]),
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["brier"] == pytest.approx(0.0625)
    assert metrics["log_loss"] == pytest.approx(0.2876820724517809)


def test_selection_uses_development_only_and_is_deterministic(
    tmp_path: Path,
) -> None:
    bundle_audit_path = prepare_fixture(tmp_path)
    report_path = tmp_path / "selection_report.json"

    first = run_moneyline_baseline_selection(
        bundle_audit_path=(bundle_audit_path),
        report_path=report_path,
        expected_season_counts={
            "20212022": 4,
            "20222023": 4,
        },
    )

    first_bytes = report_path.read_bytes()

    second = run_moneyline_baseline_selection(
        bundle_audit_path=(bundle_audit_path),
        report_path=report_path,
        expected_season_counts={
            "20212022": 4,
            "20222023": 4,
        },
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert first.model_count == 3
    assert second.already_present is True
    assert report_path.read_bytes() == (first_bytes)

    assert report["validation_loaded"] is False
    assert report["sealed_holdout_loaded"] is False
    assert all(report["structural_gates"].values())


def test_rejects_missing_primary_feature(
    tmp_path: Path,
) -> None:
    bundle_audit_path = prepare_fixture(
        tmp_path,
        feature_columns=(PRIMARY_LOGISTIC_FEATURES[:-1]),
    )

    with pytest.raises(
        MoneylineBaselineError,
        match=("Primary logistic feature set is missing columns"),
    ):
        run_moneyline_baseline_selection(
            bundle_audit_path=(bundle_audit_path),
            report_path=(tmp_path / "report.json"),
            expected_season_counts={
                "20212022": 4,
                "20222023": 4,
            },
        )
