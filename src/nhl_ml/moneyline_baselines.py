"""Leakage-safe NHL moneyline baseline selection protocol."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nhl_ml.experiment_dataset import (
    TARGET_COLUMN,
    ExperimentDatasetError,
    load_experiment_dataset,
)


class MoneylineBaselineError(ValueError):
    """Raised when a baseline experiment cannot be run safely."""


PROTOCOL_ID = "nhl_moneyline_baseline_selection_v1"

TRAIN_SEASON = "20212022"
SELECTION_SEASON = "20222023"

PRIMARY_LOGISTIC_FEATURES = (
    "home_season_to_date_games",
    "away_season_to_date_games",
    "home_minus_away_days_since_previous_game_start",
    "home_minus_away_season_to_date_win_rate",
    "home_minus_away_season_to_date_goals_for_per_game",
    "home_minus_away_season_to_date_goals_against_per_game",
    "home_minus_away_season_to_date_shots_on_goal_for_per_game",
    "home_minus_away_season_to_date_shots_on_goal_against_per_game",
    "home_minus_away_season_to_date_shot_attempt_share",
    "home_minus_away_season_to_date_shooting_percentage",
    "home_minus_away_season_to_date_save_percentage_proxy",
    "home_minus_away_season_to_date_faceoff_win_percentage",
    "home_minus_away_season_to_date_penalty_minutes_for_per_game",
    "home_minus_away_season_to_date_takeaways_for_per_game",
    "home_minus_away_season_to_date_giveaways_for_per_game",
)


@dataclass(frozen=True, slots=True)
class MoneylineBaselineResult:
    """Summary of one baseline-selection execution."""

    protocol_id: str
    train_season: str
    selection_season: str
    train_game_count: int
    selection_game_count: int
    model_count: int
    report_path: str
    report_sha256: str
    already_present: bool
    status: str


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _write_bytes_atomically(
    destination: Path,
    data: bytes,
) -> None:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_file.write(data)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
            temporary_path = Path(temporary_file.name)

        os.replace(
            temporary_path,
            destination,
        )
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _parse_utc(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise MoneylineBaselineError(f"Invalid UTC datetime: {value!r}") from exc

    if parsed.tzinfo is None:
        raise MoneylineBaselineError(f"Timezone-naive datetime: {value!r}")

    return parsed


def _feature_schema_sha256(
    feature_columns: tuple[str, ...],
) -> str:
    return _sha256(
        json.dumps(
            feature_columns,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _matrix(
    rows: list[dict[str, Any]],
    feature_columns: tuple[str, ...],
) -> np.ndarray:
    matrix = np.empty(
        (
            len(rows),
            len(feature_columns),
        ),
        dtype=float,
    )

    for row_index, row in enumerate(rows):
        for column_index, column in enumerate(feature_columns):
            if column not in row:
                raise MoneylineBaselineError(f"Feature missing from row: {column}")

            value = row[column]

            if value is None:
                matrix[
                    row_index,
                    column_index,
                ] = np.nan
                continue

            if isinstance(value, bool) or not isinstance(
                value,
                (int, float),
            ):
                raise MoneylineBaselineError(f"Feature {column} is not numeric or null: {value!r}")

            numeric_value = float(value)

            if not math.isfinite(numeric_value):
                raise MoneylineBaselineError(f"Feature {column} is non-finite: {value!r}")

            matrix[
                row_index,
                column_index,
            ] = numeric_value

    return matrix


def _targets(
    rows: list[dict[str, Any]],
) -> np.ndarray:
    targets = np.empty(
        len(rows),
        dtype=int,
    )

    for index, row in enumerate(rows):
        target = row.get(TARGET_COLUMN)

        if isinstance(target, bool) or not isinstance(target, int) or target not in {0, 1}:
            raise MoneylineBaselineError(f"Invalid binary target: {target!r}")

        targets[index] = target

    return targets


def probability_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    calibration_bins: int = 10,
) -> dict[str, float]:
    """Calculate proper scoring and calibration metrics."""
    y_true = np.asarray(
        y_true,
        dtype=int,
    )
    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    if y_true.ndim != 1 or probabilities.ndim != 1:
        raise MoneylineBaselineError("Targets and probabilities must be one-dimensional")

    if len(y_true) == 0 or len(y_true) != len(probabilities):
        raise MoneylineBaselineError("Targets and probabilities have invalid lengths")

    if not np.all(np.isin(y_true, (0, 1))):
        raise MoneylineBaselineError("Targets must be binary")

    if not np.all(np.isfinite(probabilities)):
        raise MoneylineBaselineError("Probabilities must be finite")

    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise MoneylineBaselineError("Probabilities must lie inside [0, 1]")

    clipped = np.clip(
        probabilities,
        1e-15,
        1.0 - 1e-15,
    )

    predictions = (probabilities >= 0.5).astype(int)

    accuracy = float(np.mean(predictions == y_true))
    log_loss = float(-np.mean(y_true * np.log(clipped) + (1 - y_true) * np.log(1.0 - clipped)))
    brier = float(np.mean((probabilities - y_true) ** 2))

    edges = np.linspace(
        0.0,
        1.0,
        calibration_bins + 1,
    )
    ece = 0.0

    for index in range(calibration_bins):
        lower = edges[index]
        upper = edges[index + 1]

        if index == calibration_bins - 1:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities >= lower) & (probabilities < upper)

        count = int(np.sum(mask))

        if count == 0:
            continue

        confidence = float(np.mean(probabilities[mask]))
        realised = float(np.mean(y_true[mask]))

        ece += count / len(y_true) * abs(confidence - realised)

    return {
        "accuracy": accuracy,
        "log_loss": log_loss,
        "brier": brier,
        "ece_10_bin": float(ece),
    }


def _prediction_summary(
    probabilities: np.ndarray,
) -> dict[str, float]:
    return {
        "minimum": float(np.min(probabilities)),
        "maximum": float(np.max(probabilities)),
        "mean": float(np.mean(probabilities)),
        "standard_deviation": float(np.std(probabilities)),
    }


def _evaluate_model(
    *,
    name: str,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    reference_metrics: dict[str, float] | None,
    configuration: dict[str, Any],
) -> dict[str, Any]:
    metrics = probability_metrics(
        y_true,
        probabilities,
    )

    result: dict[str, Any] = {
        "model_name": name,
        "configuration": configuration,
        "metrics": metrics,
        "prediction_summary": (_prediction_summary(probabilities)),
    }

    if reference_metrics is not None:
        result["delta_vs_train_home_rate"] = {
            "accuracy": (metrics["accuracy"] - reference_metrics["accuracy"]),
            "log_loss": (metrics["log_loss"] - reference_metrics["log_loss"]),
            "brier": (metrics["brier"] - reference_metrics["brier"]),
            "ece_10_bin": (metrics["ece_10_bin"] - reference_metrics["ece_10_bin"]),
        }

    return result


def run_moneyline_baseline_selection(
    *,
    bundle_audit_path: Path,
    report_path: Path,
    train_season: str = TRAIN_SEASON,
    selection_season: str = SELECTION_SEASON,
    expected_season_counts: dict[str, int] | None = None,
) -> MoneylineBaselineResult:
    """Run predefined baselines using development seasons only."""
    if train_season == selection_season:
        raise MoneylineBaselineError("Train and selection seasons must differ")

    if not bundle_audit_path.is_file():
        raise FileNotFoundError(f"Bundle audit does not exist: {bundle_audit_path}")

    bundle_audit_sha256 = _sha256(bundle_audit_path.read_bytes())

    try:
        dataset = load_experiment_dataset(
            bundle_audit_path=(bundle_audit_path),
            requested_splits=("development",),
            requested_seasons=(
                train_season,
                selection_season,
            ),
            include_target=True,
        )
    except (
        ExperimentDatasetError,
        FileNotFoundError,
    ) as exc:
        raise MoneylineBaselineError(f"Development dataset load failed: {exc}") from exc

    if dataset.holdout_unlocked:
        raise MoneylineBaselineError("Holdout must remain locked")

    if dataset.requested_splits != ("development",):
        raise MoneylineBaselineError("Baseline selection may load only development")

    available_features = set(dataset.feature_columns)
    missing_features = sorted(set(PRIMARY_LOGISTIC_FEATURES) - available_features)

    if missing_features:
        raise MoneylineBaselineError(
            f"Primary logistic feature set is missing columns: {missing_features}"
        )

    train_rows = [row for row in dataset.rows if row["season_id"] == train_season]
    selection_rows = [row for row in dataset.rows if row["season_id"] == selection_season]

    if not train_rows or not selection_rows:
        raise MoneylineBaselineError("Train or selection season is empty")

    observed_counts = {
        train_season: len(train_rows),
        selection_season: len(selection_rows),
    }

    if expected_season_counts is not None and observed_counts != expected_season_counts:
        raise MoneylineBaselineError(
            f"Unexpected season counts: {observed_counts}; expected {expected_season_counts}"
        )

    latest_train_start = max(_parse_utc(str(row["scheduled_start_utc"])) for row in train_rows)
    earliest_selection_start = min(
        _parse_utc(str(row["scheduled_start_utc"])) for row in selection_rows
    )

    if latest_train_start >= earliest_selection_start:
        raise MoneylineBaselineError("Train season is not strictly earlier than selection season")

    x_train = _matrix(
        train_rows,
        PRIMARY_LOGISTIC_FEATURES,
    )
    x_selection = _matrix(
        selection_rows,
        PRIMARY_LOGISTIC_FEATURES,
    )
    y_train = _targets(train_rows)
    y_selection = _targets(selection_rows)

    if set(np.unique(y_train)) != {0, 1}:
        raise MoneylineBaselineError("Training target does not contain both classes")

    if set(np.unique(y_selection)) != {0, 1}:
        raise MoneylineBaselineError("Selection target does not contain both classes")

    random_50_probabilities = np.full(
        len(y_selection),
        0.5,
        dtype=float,
    )

    train_home_win_rate = float(np.mean(y_train))
    train_home_rate_probabilities = np.full(
        len(y_selection),
        train_home_win_rate,
        dtype=float,
    )

    pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                    keep_empty_features=True,
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "logistic_regression",
                LogisticRegression(
                    C=1.0,
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=2000,
                    random_state=0,
                ),
            ),
        ]
    )

    pipeline.fit(
        x_train,
        y_train,
    )

    logistic_probabilities = pipeline.predict_proba(x_selection)[:, 1]

    home_rate_metrics = probability_metrics(
        y_selection,
        train_home_rate_probabilities,
    )

    model_results = [
        _evaluate_model(
            name="random_50",
            y_true=y_selection,
            probabilities=(random_50_probabilities),
            reference_metrics=(home_rate_metrics),
            configuration={
                "probability": 0.5,
            },
        ),
        _evaluate_model(
            name="train_home_win_rate",
            y_true=y_selection,
            probabilities=(train_home_rate_probabilities),
            reference_metrics=None,
            configuration={
                "estimated_from": train_season,
                "probability": (train_home_win_rate),
            },
        ),
        _evaluate_model(
            name="logistic_core_diff_v1",
            y_true=y_selection,
            probabilities=(logistic_probabilities),
            reference_metrics=(home_rate_metrics),
            configuration={
                "feature_count": len(PRIMARY_LOGISTIC_FEATURES),
                "features": list(PRIMARY_LOGISTIC_FEATURES),
                "imputer": ("training_median_with_missing_indicators"),
                "scaler": ("training_standard_scaler"),
                "penalty": "l2",
                "c": 1.0,
                "solver": "lbfgs",
                "max_iter": 2000,
            },
        ),
    ]

    structural_gates = {
        "passes_development_only": (dataset.requested_splits == ("development",)),
        "passes_holdout_locked": (not dataset.holdout_unlocked),
        "passes_train_before_selection": (latest_train_start < earliest_selection_start),
        "passes_expected_seasons": (
            set(dataset.season_counts)
            == {
                train_season,
                selection_season,
            }
        ),
        "passes_primary_feature_schema": (not missing_features),
        "passes_binary_train_target": (set(np.unique(y_train)) == {0, 1}),
        "passes_binary_selection_target": (set(np.unique(y_selection)) == {0, 1}),
        "passes_fixed_model_count": (len(model_results) == 3),
    }

    if not all(structural_gates.values()):
        failed = sorted(gate for gate, passed in structural_gates.items() if not passed)
        raise MoneylineBaselineError(f"Structural gates failed: {failed}")

    report = {
        "schema_version": "1.0",
        "protocol_id": PROTOCOL_ID,
        "status": ("complete_research_only_no_market"),
        "research_mode": ("research_only"),
        "market_baseline_available": False,
        "profitability_claims_allowed": False,
        "train_season": train_season,
        "selection_season": (selection_season),
        "validation_loaded": False,
        "sealed_holdout_loaded": False,
        "bundle_audit_path": str(bundle_audit_path),
        "bundle_audit_sha256": (bundle_audit_sha256),
        "source_bundle_id": (dataset.bundle_id),
        "source_dataset_id": (dataset.source_dataset_id),
        "train_game_count": len(train_rows),
        "selection_game_count": len(selection_rows),
        "train_home_win_rate": (train_home_win_rate),
        "primary_logistic_features": list(PRIMARY_LOGISTIC_FEATURES),
        "primary_logistic_feature_count": len(PRIMARY_LOGISTIC_FEATURES),
        "primary_logistic_feature_sha256": (_feature_schema_sha256(PRIMARY_LOGISTIC_FEATURES)),
        "full_bundle_feature_count": len(dataset.feature_columns),
        "models": model_results,
        "structural_gates": (structural_gates),
        "interpretation_policy": {
            "selection_result_is_not_final_validation": True,
            "validation_remains_unopened": True,
            "sealed_holdout_remains_unopened": True,
            "no_odds_loaded": True,
            "no_roi_or_profit_calculated": True,
            "negative_result_is_acceptable": True,
        },
    }

    report_data = _json_bytes(report)
    report_sha256 = _sha256(report_data)
    already_present = report_path.is_file()

    if already_present:
        if report_path.read_bytes() != report_data:
            raise MoneylineBaselineError(f"Existing baseline report differs: {report_path}")
    else:
        _write_bytes_atomically(
            report_path,
            report_data,
        )

    return MoneylineBaselineResult(
        protocol_id=PROTOCOL_ID,
        train_season=train_season,
        selection_season=(selection_season),
        train_game_count=len(train_rows),
        selection_game_count=len(selection_rows),
        model_count=len(model_results),
        report_path=str(report_path),
        report_sha256=report_sha256,
        already_present=already_present,
        status=("complete_research_only_no_market"),
    )


def result_as_dict(
    result: MoneylineBaselineResult,
) -> dict[str, Any]:
    """Convert the result to a serializable mapping."""
    return asdict(result)
