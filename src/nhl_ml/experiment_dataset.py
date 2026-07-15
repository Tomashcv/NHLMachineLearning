"""Build and load an audited NHL experimental dataset bundle."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


class ExperimentDatasetError(ValueError):
    """Raised when an experimental dataset fails validation."""


class SealedHoldoutError(ExperimentDatasetError):
    """Raised when sealed-holdout access is not explicitly unlocked."""


BUNDLE_ID = "nhl_regular_season_2021_2025_experiment_bundle_v1"

ALLOWED_SPLITS = (
    "development",
    "validation",
    "sealed_holdout",
)

SEALED_HOLDOUT_SPLIT = "sealed_holdout"

HOLDOUT_UNLOCK_TOKEN = "UNLOCK_20242025_SEALED_HOLDOUT_AFTER_PROTOCOL_FREEZE"

TARGET_COLUMN = "target_home_win"

SANITIZED_METADATA_COLUMNS = (
    "schema_version",
    "dataset_id",
    "game_id",
    "season_id",
    "split_role",
    "scheduled_start_utc",
    "home_team_id",
    "away_team_id",
)


@dataclass(frozen=True, slots=True)
class ExperimentBundleResult:
    """Summary of a split-bundle build."""

    bundle_id: str
    source_dataset_id: str
    game_count: int
    split_counts: dict[str, int]
    output_dir: str
    audit_path: str
    already_present: bool
    status: str


@dataclass(frozen=True, slots=True)
class LoadedExperimentDataset:
    """Rows and schema loaded for one experimental stage."""

    bundle_id: str
    source_dataset_id: str
    requested_splits: tuple[str, ...]
    requested_seasons: tuple[str, ...] | None
    row_count: int
    feature_columns: tuple[str, ...]
    metadata_columns: tuple[str, ...]
    target_column: str | None
    split_counts: dict[str, int]
    season_counts: dict[str, int]
    rows: tuple[dict[str, Any], ...]
    holdout_unlocked: bool


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ExperimentDatasetError(f"Invalid JSON file: {path}") from exc

    if not isinstance(payload, dict):
        raise ExperimentDatasetError(f"Expected a JSON object in {path}")

    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            stripped = line.strip()

            if not stripped:
                continue

            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ExperimentDatasetError(
                    f"Invalid JSONL in {path} at line {line_number}"
                ) from exc

            if not isinstance(row, dict):
                raise ExperimentDatasetError(f"Expected an object in {path} at line {line_number}")

            rows.append(row)

    return rows


def _serialize_jsonl(
    rows: list[dict[str, Any]],
) -> bytes:
    text = "\n".join(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for row in rows
    )

    if text:
        text += "\n"

    return text.encode("utf-8")


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


def _write_json_atomically(
    destination: Path,
    payload: dict[str, Any],
) -> None:
    data = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")

    _write_bytes_atomically(
        destination,
        data,
    )


def _parse_utc(value: Any) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ExperimentDatasetError(f"Invalid scheduled_start_utc: {value!r}")

    normalized = value.strip().replace(
        "Z",
        "+00:00",
    )

    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ExperimentDatasetError(f"Invalid datetime: {value!r}") from exc

    if parsed.tzinfo is None:
        raise ExperimentDatasetError(f"Timezone-naive datetime: {value!r}")

    return parsed


def _validate_feature_value(
    *,
    row: dict[str, Any],
    feature: str,
) -> None:
    if feature not in row:
        raise ExperimentDatasetError(f"Feature is missing: {feature}")

    value = row[feature]

    if value is None:
        return

    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise ExperimentDatasetError(f"Feature {feature} is not numeric or null: {value!r}")

    if not math.isfinite(float(value)):
        raise ExperimentDatasetError(f"Feature {feature} is non-finite: {value!r}")


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


def _verify_model_ready_source(
    *,
    source_path: Path,
    source_audit_path: Path,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    str,
    tuple[str, ...],
]:
    if not source_path.is_file():
        raise FileNotFoundError(f"Model-ready source does not exist: {source_path}")

    if not source_audit_path.is_file():
        raise FileNotFoundError(f"Model-ready audit does not exist: {source_audit_path}")

    source_data = source_path.read_bytes()
    source_sha256 = _sha256(source_data)
    source_audit = _read_json(source_audit_path)

    if source_audit.get("status") != "complete":
        raise ExperimentDatasetError("Model-ready source audit is not complete")

    if str(source_audit.get("output_sha256", "")) != source_sha256:
        raise ExperimentDatasetError("Model-ready source hash does not match its audit")

    feature_columns_raw = source_audit.get("feature_columns")

    if not isinstance(feature_columns_raw, list):
        raise ExperimentDatasetError("Model-ready audit has no feature list")

    feature_columns = tuple(str(column) for column in feature_columns_raw)

    if not feature_columns:
        raise ExperimentDatasetError("Model-ready feature list is empty")

    if len(set(feature_columns)) != len(feature_columns):
        raise ExperimentDatasetError("Model-ready feature list has duplicates")

    if any(column.startswith("target_") for column in feature_columns):
        raise ExperimentDatasetError("A target column appears in the feature schema")

    expected_feature_sha256 = str(
        source_audit.get(
            "feature_columns_sha256",
            "",
        )
    )
    actual_feature_sha256 = _feature_schema_sha256(feature_columns)

    if expected_feature_sha256 != actual_feature_sha256:
        raise ExperimentDatasetError("Feature-schema hash does not match")

    target_columns = source_audit.get(
        "target_columns",
        [],
    )

    if TARGET_COLUMN not in target_columns:
        raise ExperimentDatasetError(f"{TARGET_COLUMN} is absent from the target schema")

    rows = _read_jsonl(source_path)

    expected_count = int(source_audit.get("game_count", 0))

    if len(rows) != expected_count:
        raise ExperimentDatasetError(
            f"Model-ready source has {len(rows)} rows, expected {expected_count}"
        )

    return (
        rows,
        source_audit,
        source_sha256,
        feature_columns,
    )


def _sanitize_row(
    *,
    row: dict[str, Any],
    feature_columns: tuple[str, ...],
    source_dataset_id: str,
) -> dict[str, Any]:
    if str(row.get("schema_version")) != "1.0":
        raise ExperimentDatasetError("Expected model-ready schema version 1.0")

    if str(row.get("dataset_id")) != source_dataset_id:
        raise ExperimentDatasetError("Model-ready dataset ID mismatch")

    game_id = str(row.get("game_id", "")).strip()
    season_id = str(row.get("season_id", "")).strip()
    split_role = str(row.get("split_role", "")).strip()
    home_team_id = str(row.get("home_team_id", "")).strip()
    away_team_id = str(row.get("away_team_id", "")).strip()
    scheduled_start_utc = str(row.get("scheduled_start_utc", "")).strip()

    if not all(
        (
            game_id,
            season_id,
            home_team_id,
            away_team_id,
            scheduled_start_utc,
        )
    ):
        raise ExperimentDatasetError("Model-ready row has empty identifiers")

    if split_role not in ALLOWED_SPLITS:
        raise ExperimentDatasetError(f"Unexpected split role: {split_role!r}")

    if home_team_id == away_team_id:
        raise ExperimentDatasetError(f"Identical teams in game {game_id}")

    _parse_utc(scheduled_start_utc)

    target = row.get(TARGET_COLUMN)

    if isinstance(target, bool) or not isinstance(target, int) or target not in {0, 1}:
        raise ExperimentDatasetError(f"Invalid binary target for game {game_id}: {target!r}")

    sanitized: dict[str, Any] = {
        "schema_version": "1.0",
        "dataset_id": source_dataset_id,
        "game_id": game_id,
        "season_id": season_id,
        "split_role": split_role,
        "scheduled_start_utc": (scheduled_start_utc),
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
    }

    for feature in feature_columns:
        _validate_feature_value(
            row=row,
            feature=feature,
        )
        sanitized[feature] = row[feature]

    sanitized[TARGET_COLUMN] = target

    return sanitized


def build_experiment_dataset_bundle(
    *,
    source_path: Path,
    source_audit_path: Path,
    output_dir: Path,
    audit_path: Path,
) -> ExperimentBundleResult:
    """Create separate audited files for each frozen split."""
    (
        source_rows,
        source_audit,
        source_sha256,
        feature_columns,
    ) = _verify_model_ready_source(
        source_path=source_path,
        source_audit_path=source_audit_path,
    )

    source_dataset_id = str(source_audit.get("dataset_id", "")).strip()

    if not source_dataset_id:
        raise ExperimentDatasetError("Model-ready audit has no dataset ID")

    rows_by_split: dict[
        str,
        list[dict[str, Any]],
    ] = {split_role: [] for split_role in ALLOWED_SPLITS}

    game_ids: set[str] = set()

    for source_row in source_rows:
        row = _sanitize_row(
            row=source_row,
            feature_columns=feature_columns,
            source_dataset_id=source_dataset_id,
        )

        game_id = row["game_id"]

        if game_id in game_ids:
            raise ExperimentDatasetError(f"Duplicate game ID: {game_id}")

        game_ids.add(game_id)
        rows_by_split[row["split_role"]].append(row)

    for split_rows in rows_by_split.values():
        split_rows.sort(
            key=lambda row: (
                _parse_utc(row["scheduled_start_utc"]),
                row["game_id"],
            )
        )

    actual_split_counts = {
        split_role: len(rows_by_split[split_role]) for split_role in ALLOWED_SPLITS
    }

    expected_split_counts_raw = source_audit.get(
        "split_game_counts",
        {},
    )

    expected_split_counts = {
        split_role: int(
            expected_split_counts_raw.get(
                split_role,
                0,
            )
        )
        for split_role in ALLOWED_SPLITS
    }

    gates = {
        "passes_source_game_count": (
            len(game_ids)
            == int(
                source_audit.get(
                    "game_count",
                    0,
                )
            )
        ),
        "passes_unique_game_ids": (len(game_ids) == len(source_rows)),
        "passes_split_counts": (actual_split_counts == expected_split_counts),
        "passes_all_splits_nonempty": all(
            actual_split_counts[split_role] > 0 for split_role in ALLOWED_SPLITS
        ),
        "passes_no_target_in_features": (
            not any(column.startswith("target_") for column in feature_columns)
        ),
    }

    failed_gates = sorted(gate for gate, passed in gates.items() if not passed)

    if failed_gates:
        raise ExperimentDatasetError(f"Experiment bundle gates failed: {failed_gates}")

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    split_files: dict[str, dict[str, Any]] = {}
    all_already_present = True

    for split_role in ALLOWED_SPLITS:
        split_path = output_dir / f"{split_role}.jsonl"
        split_data = _serialize_jsonl(rows_by_split[split_role])
        split_sha256 = _sha256(split_data)

        if split_path.is_file():
            if split_path.read_bytes() != split_data:
                raise ExperimentDatasetError(f"Existing split output differs: {split_path}")
        else:
            all_already_present = False
            _write_bytes_atomically(
                split_path,
                split_data,
            )

        split_files[split_role] = {
            "path": str(split_path),
            "row_count": len(rows_by_split[split_role]),
            "sha256": split_sha256,
        }

    bundle_audit = {
        "schema_version": "1.0",
        "bundle_id": BUNDLE_ID,
        "status": "complete",
        "source_dataset_id": (source_dataset_id),
        "source_path": str(source_path),
        "source_audit_path": str(source_audit_path),
        "source_sha256": source_sha256,
        "source_game_count": len(source_rows),
        "feature_columns": list(feature_columns),
        "feature_column_count": len(feature_columns),
        "feature_columns_sha256": (_feature_schema_sha256(feature_columns)),
        "metadata_columns": list(SANITIZED_METADATA_COLUMNS),
        "target_column": TARGET_COLUMN,
        "removed_postgame_columns": [
            "target_home_final_score",
            "target_away_final_score",
            "target_outcome_type",
        ],
        "default_loader_splits": ["development"],
        "sealed_holdout_policy": {
            "split_role": (SEALED_HOLDOUT_SPLIT),
            "requires_explicit_flag": True,
            "requires_unlock_token": True,
        },
        "split_counts": (actual_split_counts),
        "split_files": split_files,
        "gates": gates,
    }

    _write_json_atomically(
        audit_path,
        bundle_audit,
    )

    return ExperimentBundleResult(
        bundle_id=BUNDLE_ID,
        source_dataset_id=(source_dataset_id),
        game_count=len(source_rows),
        split_counts=actual_split_counts,
        output_dir=str(output_dir),
        audit_path=str(audit_path),
        already_present=(all_already_present),
        status="complete",
    )


def load_experiment_dataset(
    *,
    bundle_audit_path: Path,
    requested_splits: tuple[str, ...] = ("development",),
    requested_seasons: tuple[str, ...] | None = None,
    include_target: bool = True,
    allow_sealed_holdout: bool = False,
    holdout_unlock_token: str | None = None,
) -> LoadedExperimentDataset:
    """Load only explicitly requested and permitted split files."""
    if not requested_splits:
        raise ExperimentDatasetError("At least one split must be requested")

    requested_splits = tuple(dict.fromkeys(requested_splits))

    unknown_splits = set(requested_splits) - set(ALLOWED_SPLITS)

    if unknown_splits:
        raise ExperimentDatasetError(f"Unknown requested splits: {sorted(unknown_splits)}")

    holdout_requested = SEALED_HOLDOUT_SPLIT in requested_splits

    if holdout_requested and (
        not allow_sealed_holdout or holdout_unlock_token != HOLDOUT_UNLOCK_TOKEN
    ):
        raise SealedHoldoutError(
            "The sealed holdout remains locked. "
            "Both the explicit flag and exact "
            "unlock token are required."
        )

    audit = _read_json(bundle_audit_path)

    if audit.get("status") != "complete":
        raise ExperimentDatasetError("Experiment bundle audit is not complete")

    feature_columns_raw = audit.get("feature_columns")

    if not isinstance(
        feature_columns_raw,
        list,
    ):
        raise ExperimentDatasetError("Bundle audit has no feature list")

    feature_columns = tuple(str(column) for column in feature_columns_raw)

    if _feature_schema_sha256(feature_columns) != str(
        audit.get(
            "feature_columns_sha256",
            "",
        )
    ):
        raise ExperimentDatasetError("Bundle feature-schema hash mismatch")

    split_files = audit.get("split_files")

    if not isinstance(split_files, dict):
        raise ExperimentDatasetError("Bundle audit has no split files")

    requested_season_set = None if requested_seasons is None else set(requested_seasons)

    loaded_rows: list[dict[str, Any]] = []
    seen_game_ids: set[str] = set()

    for split_role in requested_splits:
        split_metadata = split_files.get(split_role)

        if not isinstance(
            split_metadata,
            dict,
        ):
            raise ExperimentDatasetError(f"Missing split metadata: {split_role}")

        split_path = Path(str(split_metadata.get("path", "")))

        if not split_path.is_file():
            raise FileNotFoundError(f"Requested split does not exist: {split_path}")

        split_data = split_path.read_bytes()

        if _sha256(split_data) != str(
            split_metadata.get(
                "sha256",
                "",
            )
        ):
            raise ExperimentDatasetError(f"Split hash mismatch: {split_role}")

        split_rows = _read_jsonl(split_path)

        if len(split_rows) != int(
            split_metadata.get(
                "row_count",
                -1,
            )
        ):
            raise ExperimentDatasetError(f"Split row-count mismatch: {split_role}")

        for source_row in split_rows:
            if (
                str(
                    source_row.get(
                        "split_role",
                        "",
                    )
                )
                != split_role
            ):
                raise ExperimentDatasetError(f"Row stored in wrong split: {split_role}")

            season_id = str(
                source_row.get(
                    "season_id",
                    "",
                )
            )

            if requested_season_set is not None and season_id not in requested_season_set:
                continue

            game_id = str(
                source_row.get(
                    "game_id",
                    "",
                )
            )

            if game_id in seen_game_ids:
                raise ExperimentDatasetError(f"Duplicate loaded game ID: {game_id}")

            seen_game_ids.add(game_id)

            returned_row = {
                column: source_row[column]
                for column in (
                    *SANITIZED_METADATA_COLUMNS,
                    *feature_columns,
                )
            }

            for feature in feature_columns:
                _validate_feature_value(
                    row=returned_row,
                    feature=feature,
                )

            if include_target:
                target = source_row.get(TARGET_COLUMN)

                if (
                    isinstance(target, bool)
                    or not isinstance(
                        target,
                        int,
                    )
                    or target not in {0, 1}
                ):
                    raise ExperimentDatasetError(f"Invalid loaded target: {target!r}")

                returned_row[TARGET_COLUMN] = target

            loaded_rows.append(returned_row)

    loaded_rows.sort(
        key=lambda row: (
            _parse_utc(row["scheduled_start_utc"]),
            row["game_id"],
        )
    )

    if requested_season_set is not None and not loaded_rows:
        raise ExperimentDatasetError("No rows matched the requested seasons")

    split_counts = Counter(row["split_role"] for row in loaded_rows)
    season_counts = Counter(row["season_id"] for row in loaded_rows)

    return LoadedExperimentDataset(
        bundle_id=str(audit["bundle_id"]),
        source_dataset_id=str(audit["source_dataset_id"]),
        requested_splits=(requested_splits),
        requested_seasons=(requested_seasons),
        row_count=len(loaded_rows),
        feature_columns=(feature_columns),
        metadata_columns=(SANITIZED_METADATA_COLUMNS),
        target_column=(TARGET_COLUMN if include_target else None),
        split_counts=dict(sorted(split_counts.items())),
        season_counts=dict(sorted(season_counts.items())),
        rows=tuple(loaded_rows),
        holdout_unlocked=(holdout_requested),
    )


def result_as_dict(
    result: ExperimentBundleResult,
) -> dict[str, Any]:
    """Convert a bundle result to a mapping."""
    return asdict(result)
