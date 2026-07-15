"""Build and verify deterministic season-scale NHL PBP batch configs."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


class SeasonScalePbpBatchError(ValueError):
    """Raised when a season-scale PBP batch config is invalid."""


@dataclass(frozen=True, slots=True)
class SeasonScalePbpBatchResult:
    """Summary of the season-scale PBP batch-config build."""

    corpus_id: str
    season_count: int
    game_count: int
    config_paths: list[str]
    config_sha256s: dict[str, str]
    subset_sha256s: dict[str, str]
    already_present_count: int
    status: str


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SeasonScalePbpBatchError(f"Invalid JSON file: {path}") from exc

    if not isinstance(payload, dict):
        raise SeasonScalePbpBatchError(f"Expected JSON object in {path}")

    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()

            if not stripped:
                continue

            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise SeasonScalePbpBatchError(
                    f"Invalid JSONL in {path} at line {line_number}"
                ) from exc

            if not isinstance(record, dict):
                raise SeasonScalePbpBatchError(f"Expected object in {path} at line {line_number}")

            records.append(record)

    return records


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise SeasonScalePbpBatchError(f"Invalid YAML file: {path}") from exc

    if not isinstance(payload, dict):
        raise SeasonScalePbpBatchError(f"Expected YAML mapping in {path}")

    return payload


def _write_bytes_atomically(
    destination: Path,
    data: bytes,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
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

        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _selection_record(row: dict[str, Any]) -> dict[str, Any]:
    """Return only stable, selection-critical inventory fields."""
    return {
        "game_id": str(row["game_id"]),
        "season_id": str(row["season_id"]),
        "split_role": str(row["split_role"]),
        "scheduled_start_utc": str(row["scheduled_start_utc"]),
        "expected_outcome": str(row["expected_outcome"]),
        "away_team_id": str(row["away_team_id"]),
        "home_team_id": str(row["home_team_id"]),
        "source_filename": str(row["source_filename"]),
        "source_sha256": str(row["source_sha256"]),
    }


def _subset_data(rows: list[dict[str, Any]]) -> bytes:
    selection_rows = [_selection_record(row) for row in rows]
    selection_rows.sort(
        key=lambda row: (
            row["scheduled_start_utc"],
            row["game_id"],
        )
    )

    serialized = "\n".join(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for row in selection_rows
    )

    if serialized:
        serialized += "\n"

    return serialized.encode("utf-8")


def _config_payload(
    *,
    corpus_id: str,
    season_id: str,
    split_role: str,
    source_inventory_sha256: str,
    season_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], str]:
    season_rows = sorted(
        season_rows,
        key=lambda row: (
            row["scheduled_start_utc"],
            row["game_id"],
        ),
    )

    subset_sha256 = _sha256(_subset_data(season_rows))
    batch_id = f"nhl_regular_season_{season_id}_pbp_v1"

    payload = {
        "schema_version": "1.0",
        "batch": {
            "batch_id": batch_id,
            "expected_game_count": len(season_rows),
            "source_selection_id": f"{corpus_id}:{season_id}",
            "source_selection_sha256": subset_sha256,
            "source_inventory_sha256": source_inventory_sha256,
            "corpus_id": corpus_id,
            "season_id": season_id,
            "split_role": split_role,
        },
        "games": [
            {
                "game_id": str(row["game_id"]),
                "expected_outcome": str(row["expected_outcome"]),
                "source_filename": str(row["source_filename"]),
                "scheduled_start_utc": str(row["scheduled_start_utc"]),
                "away_team_id": str(row["away_team_id"]),
                "home_team_id": str(row["home_team_id"]),
                "source_sha256": str(row["source_sha256"]),
            }
            for row in season_rows
        ],
    }

    return payload, subset_sha256


def build_season_scale_pbp_batch_configs(
    *,
    inventory_path: Path,
    inventory_audit_path: Path,
    output_dir: Path,
) -> SeasonScalePbpBatchResult:
    """Build one exact PBP config per season."""
    if not inventory_path.is_file():
        raise FileNotFoundError(f"Verified inventory does not exist: {inventory_path}")

    if not inventory_audit_path.is_file():
        raise FileNotFoundError(f"Inventory audit does not exist: {inventory_audit_path}")

    inventory_data = inventory_path.read_bytes()
    inventory_sha256 = _sha256(inventory_data)
    inventory_audit = _read_json(inventory_audit_path)

    if inventory_audit.get("status") != "complete":
        raise SeasonScalePbpBatchError("Verified inventory audit is not complete")

    if str(inventory_audit.get("output_sha256", "")) != inventory_sha256:
        raise SeasonScalePbpBatchError("Verified inventory hash does not match its audit")

    corpus_id = str(inventory_audit.get("corpus_id", "")).strip()

    if not corpus_id:
        raise SeasonScalePbpBatchError("Verified inventory corpus ID is missing")

    inventory_rows = _read_jsonl(inventory_path)
    expected_game_count = int(inventory_audit.get("game_count", 0))

    if len(inventory_rows) != expected_game_count:
        raise SeasonScalePbpBatchError(
            f"Inventory contains {len(inventory_rows)} rows, expected {expected_game_count}"
        )

    rows_by_season: dict[str, list[dict[str, Any]]] = {}
    split_roles_by_season: dict[str, set[str]] = {}

    for row in inventory_rows:
        season_id = str(row["season_id"])
        rows_by_season.setdefault(season_id, []).append(row)
        split_roles_by_season.setdefault(season_id, set()).add(str(row["split_role"]))

    expected_season_counts = {
        str(key): int(value)
        for key, value in inventory_audit.get(
            "season_counts",
            {},
        ).items()
    }

    if {
        season_id: len(rows) for season_id, rows in rows_by_season.items()
    } != expected_season_counts:
        raise SeasonScalePbpBatchError("Inventory season counts do not match its audit")

    config_paths: list[str] = []
    config_sha256s: dict[str, str] = {}
    subset_sha256s: dict[str, str] = {}
    already_present_count = 0

    for season_id in sorted(rows_by_season):
        split_roles = split_roles_by_season[season_id]

        if len(split_roles) != 1:
            raise SeasonScalePbpBatchError(
                f"Season {season_id} has multiple split roles: {sorted(split_roles)}"
            )

        split_role = next(iter(split_roles))
        season_rows = rows_by_season[season_id]

        payload, subset_sha256 = _config_payload(
            corpus_id=corpus_id,
            season_id=season_id,
            split_role=split_role,
            source_inventory_sha256=inventory_sha256,
            season_rows=season_rows,
        )

        config_data = yaml.safe_dump(
            payload,
            allow_unicode=True,
            sort_keys=False,
        ).encode("utf-8")

        config_path = output_dir / f"pbp_season_{season_id}.yaml"
        already_present = config_path.is_file()

        if already_present:
            if config_path.read_bytes() != config_data:
                raise SeasonScalePbpBatchError(f"Existing config differs: {config_path}")

            already_present_count += 1
        else:
            _write_bytes_atomically(config_path, config_data)

        config_paths.append(str(config_path))
        config_sha256s[season_id] = _sha256(config_data)
        subset_sha256s[season_id] = subset_sha256

    return SeasonScalePbpBatchResult(
        corpus_id=corpus_id,
        season_count=len(rows_by_season),
        game_count=len(inventory_rows),
        config_paths=config_paths,
        config_sha256s=config_sha256s,
        subset_sha256s=subset_sha256s,
        already_present_count=already_present_count,
        status="complete",
    )


def verify_season_pbp_batch_config(
    *,
    season_id: str,
    config_path: Path,
    inventory_path: Path,
    inventory_audit_path: Path,
) -> dict[str, Any]:
    """Verify a season config against the frozen verified inventory."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Season config does not exist: {config_path}")

    if not inventory_path.is_file():
        raise FileNotFoundError(f"Verified inventory does not exist: {inventory_path}")

    inventory_data = inventory_path.read_bytes()
    inventory_sha256 = _sha256(inventory_data)
    inventory_audit = _read_json(inventory_audit_path)

    if inventory_audit.get("status") != "complete":
        raise SeasonScalePbpBatchError("Verified inventory audit is not complete")

    if str(inventory_audit.get("output_sha256", "")) != inventory_sha256:
        raise SeasonScalePbpBatchError("Verified inventory hash does not match its audit")

    season_rows = [row for row in _read_jsonl(inventory_path) if str(row["season_id"]) == season_id]

    if not season_rows:
        raise SeasonScalePbpBatchError(f"No verified inventory rows for {season_id}")

    split_roles = {str(row["split_role"]) for row in season_rows}

    if len(split_roles) != 1:
        raise SeasonScalePbpBatchError(f"Season {season_id} has multiple split roles")

    expected_payload, expected_subset_sha256 = _config_payload(
        corpus_id=str(inventory_audit["corpus_id"]),
        season_id=season_id,
        split_role=next(iter(split_roles)),
        source_inventory_sha256=inventory_sha256,
        season_rows=season_rows,
    )
    actual_payload = _read_yaml(config_path)

    if actual_payload != expected_payload:
        raise SeasonScalePbpBatchError(
            f"Season config does not match verified inventory: {config_path}"
        )

    game_ids = [str(game["game_id"]) for game in actual_payload["games"]]
    outcome_counts = Counter(str(game["expected_outcome"]) for game in actual_payload["games"])

    return {
        "season_id": season_id,
        "batch_id": str(actual_payload["batch"]["batch_id"]),
        "split_role": str(actual_payload["batch"]["split_role"]),
        "game_count": len(game_ids),
        "unique_game_count": len(set(game_ids)),
        "subset_sha256": expected_subset_sha256,
        "config_sha256": _sha256(config_path.read_bytes()),
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "status": "verified",
    }


def result_as_dict(
    result: SeasonScalePbpBatchResult,
) -> dict[str, Any]:
    """Convert the batch-config result into a mapping."""
    return asdict(result)
