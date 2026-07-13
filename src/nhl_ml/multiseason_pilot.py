"""Deterministic multiseason NHL pilot selection."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

from nhl_ml.config import ConfigError, load_yaml


class MultiseasonPilotError(ValueError):
    """Raised when the multiseason pilot cannot be selected safely."""


@dataclass(frozen=True, slots=True)
class PilotSelectionConfig:
    """Frozen rules for selecting the multiseason pilot."""

    selection_id: str
    target_games: int
    maximum_games_per_source_date: int
    minimum_playoff_games: int
    required_outcomes: tuple[str, ...]
    season_quotas: dict[str, int]

    @classmethod
    def from_path(cls, path: Path) -> PilotSelectionConfig:
        payload = load_yaml(path)

        try:
            selection = payload["selection"]
            required_outcomes = tuple(payload["required_outcomes"])
            season_quotas = {
                str(season_id): int(quota) for season_id, quota in payload["season_quotas"].items()
            }

            config = cls(
                selection_id=str(selection["selection_id"]).strip(),
                target_games=int(selection["target_games"]),
                maximum_games_per_source_date=int(selection["maximum_games_per_source_date"]),
                minimum_playoff_games=int(selection["minimum_playoff_games"]),
                required_outcomes=required_outcomes,
                season_quotas=season_quotas,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigError(f"Invalid pilot selection configuration: {exc}") from exc

        config.validate()
        return config

    def validate(self) -> None:
        if not self.selection_id:
            raise ConfigError("selection_id cannot be empty")

        if self.target_games <= 0:
            raise ConfigError("target_games must be positive")

        if self.maximum_games_per_source_date <= 0:
            raise ConfigError("maximum_games_per_source_date must be positive")

        if self.minimum_playoff_games < 0:
            raise ConfigError("minimum_playoff_games cannot be negative")

        if not self.season_quotas:
            raise ConfigError("season_quotas cannot be empty")

        if any(quota <= 0 for quota in self.season_quotas.values()):
            raise ConfigError("Every season quota must be positive")

        if sum(self.season_quotas.values()) != self.target_games:
            raise ConfigError("Season quotas must sum exactly to target_games")

        allowed_outcomes = {
            "regulation",
            "overtime_only",
            "shootout",
        }
        unknown = sorted(set(self.required_outcomes) - allowed_outcomes)

        if unknown:
            raise ConfigError(f"Unknown required outcomes: {unknown}")


@dataclass(frozen=True, slots=True)
class MultiseasonPilotResult:
    """Summary of a completed pilot selection."""

    selection_id: str
    selected_games_path: str
    audit_path: str
    selected_game_count: int
    season_count: int
    regular_season_count: int
    playoff_count: int
    regulation_count: int
    overtime_only_count: int
    shootout_count: int
    selection_sha256: str
    status: str


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_bytes_atomically(destination: Path, data: bytes) -> None:
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
                raise MultiseasonPilotError(
                    f"Invalid JSONL in {path} at line {line_number}"
                ) from exc

            if not isinstance(record, dict):
                raise MultiseasonPilotError(f"Expected JSON object in {path} at line {line_number}")

            records.append(record)

    return records


def _discover_records(canonical_root: Path) -> list[dict[str, Any]]:
    files = sorted(canonical_root.glob("*/games_*.jsonl"))

    if not files:
        raise MultiseasonPilotError(f"No canonical files found under {canonical_root}")

    records_by_game_id: dict[str, dict[str, Any]] = {}

    for path in files:
        for record in _read_jsonl(path):
            game = record.get("game")

            if not isinstance(game, dict):
                raise MultiseasonPilotError(f"Canonical record in {path} has no game object")

            game_id = str(game.get("game_id", "")).strip()

            if not game_id:
                raise MultiseasonPilotError(f"Canonical record in {path} has no game_id")

            existing = records_by_game_id.get(game_id)

            if existing is not None and existing != record:
                raise MultiseasonPilotError(f"Conflicting canonical records for game {game_id}")

            records_by_game_id[game_id] = record

    return sorted(
        records_by_game_id.values(),
        key=lambda record: (
            record["game"]["scheduled_start_utc"],
            record["game"]["game_id"],
        ),
    )


def _source_date(record: dict[str, Any]) -> str:
    raw_relative_path = record.get("source_raw_relative_path")

    if not isinstance(raw_relative_path, str):
        raise MultiseasonPilotError("Canonical record is missing source_raw_relative_path")

    for part in Path(raw_relative_path).parts:
        try:
            parsed = date.fromisoformat(part)
        except ValueError:
            continue

        return parsed.isoformat()

    raise MultiseasonPilotError(f"Could not determine source date from {raw_relative_path!r}")


def _outcome_type(record: dict[str, Any]) -> str:
    game = record["game"]

    if game.get("status") != "final":
        return "not_final"

    if game.get("went_to_shootout") is True:
        return "shootout"

    if game.get("went_to_overtime") is True:
        return "overtime_only"

    return "regulation"


def _serialize_jsonl(records: list[dict[str, Any]]) -> bytes:
    lines = [
        json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for record in records
    ]

    serialized = "\n".join(lines)

    if serialized:
        serialized += "\n"

    return serialized.encode("utf-8")


def select_multiseason_pilot(
    *,
    canonical_root: Path,
    config: PilotSelectionConfig,
    output_path: Path,
    audit_path: Path,
) -> MultiseasonPilotResult:
    """Select a deterministic, quota-controlled multiseason pilot."""
    all_records = _discover_records(canonical_root)

    eligible = [
        record
        for record in all_records
        if record["game"].get("status") == "final"
        and record["game"].get("season_id") in config.season_quotas
    ]

    available_by_season = Counter(record["game"]["season_id"] for record in eligible)

    insufficient = {
        season_id: {
            "required": quota,
            "available": available_by_season[season_id],
        }
        for season_id, quota in config.season_quotas.items()
        if available_by_season[season_id] < quota
    }

    if insufficient:
        raise MultiseasonPilotError(f"Insufficient games for season quotas: {insufficient}")

    selected: dict[str, dict[str, Any]] = {}
    selection_reasons: dict[str, set[str]] = {}
    season_counts: Counter[str] = Counter()
    source_date_counts: Counter[str] = Counter()

    def can_add(record: dict[str, Any]) -> bool:
        game = record["game"]
        season_id = game["season_id"]
        source_date = _source_date(record)

        if game["game_id"] in selected:
            return True

        if season_counts[season_id] >= config.season_quotas[season_id]:
            return False

        return source_date_counts[source_date] < config.maximum_games_per_source_date

    def add_record(record: dict[str, Any], reason: str) -> None:
        game = record["game"]
        game_id = game["game_id"]

        if game_id in selected:
            selection_reasons[game_id].add(reason)
            return

        if not can_add(record):
            raise MultiseasonPilotError(f"Selection constraints rejected required game {game_id}")

        selected[game_id] = record
        selection_reasons[game_id] = {reason}
        season_counts[game["season_id"]] += 1
        source_date_counts[_source_date(record)] += 1

    playoff_candidates = [
        record for record in eligible if record["game"].get("game_type") == "playoff"
    ]

    if len(playoff_candidates) < config.minimum_playoff_games:
        raise MultiseasonPilotError("Not enough playoff games for the required minimum")

    for record in playoff_candidates[: config.minimum_playoff_games]:
        add_record(record, "required_playoff")

    for required_outcome in config.required_outcomes:
        existing = next(
            (record for record in selected.values() if _outcome_type(record) == required_outcome),
            None,
        )

        if existing is not None:
            add_record(existing, f"required_outcome:{required_outcome}")
            continue

        candidate = next(
            (
                record
                for record in eligible
                if _outcome_type(record) == required_outcome and can_add(record)
            ),
            None,
        )

        if candidate is None:
            raise MultiseasonPilotError(f"No eligible game for required outcome {required_outcome}")

        add_record(candidate, f"required_outcome:{required_outcome}")

    for season_id, quota in config.season_quotas.items():
        season_candidates = [
            record for record in eligible if record["game"]["season_id"] == season_id
        ]

        for record in season_candidates:
            if season_counts[season_id] >= quota:
                break

            if record["game"]["game_id"] in selected:
                continue

            if not can_add(record):
                continue

            add_record(record, f"season_quota:{season_id}")

        if season_counts[season_id] != quota:
            raise MultiseasonPilotError(
                f"Could not satisfy quota for season {season_id}: "
                f"{season_counts[season_id]}/{quota}"
            )

    if len(selected) != config.target_games:
        raise MultiseasonPilotError(
            f"Selected {len(selected)} games, expected {config.target_games}"
        )

    selected_records: list[dict[str, Any]] = []

    for record in sorted(
        selected.values(),
        key=lambda item: (
            item["game"]["scheduled_start_utc"],
            item["game"]["game_id"],
        ),
    ):
        game_id = record["game"]["game_id"]
        output_record = dict(record)
        output_record["pilot_selection"] = {
            "selection_id": config.selection_id,
            "source_date": _source_date(record),
            "reasons": sorted(selection_reasons[game_id]),
        }
        selected_records.append(output_record)

    output_data = _serialize_jsonl(selected_records)
    selection_digest = _sha256(output_data)

    _write_bytes_atomically(output_path, output_data)

    game_type_counts = Counter(record["game"]["game_type"] for record in selected_records)
    outcome_counts = Counter(_outcome_type(record) for record in selected_records)

    audit = {
        "schema_version": "1.0",
        "selection_id": config.selection_id,
        "status": "complete",
        "selected_game_count": len(selected_records),
        "selection_sha256": selection_digest,
        "season_quotas": config.season_quotas,
        "season_counts": dict(sorted(season_counts.items())),
        "source_date_counts": dict(sorted(source_date_counts.items())),
        "maximum_games_per_source_date": (config.maximum_games_per_source_date),
        "game_type_counts": dict(sorted(game_type_counts.items())),
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "selected_game_ids": [record["game"]["game_id"] for record in selected_records],
        "passes_target_count": (len(selected_records) == config.target_games),
        "passes_season_quotas": all(
            season_counts[season_id] == quota for season_id, quota in config.season_quotas.items()
        ),
        "passes_source_date_cap": all(
            count <= config.maximum_games_per_source_date for count in source_date_counts.values()
        ),
        "passes_playoff_minimum": (game_type_counts["playoff"] >= config.minimum_playoff_games),
        "passes_required_outcomes": all(
            outcome_counts[outcome] > 0 for outcome in config.required_outcomes
        ),
    }

    audit_data = (json.dumps(audit, indent=2, sort_keys=True) + "\n").encode("utf-8")

    _write_bytes_atomically(audit_path, audit_data)

    return MultiseasonPilotResult(
        selection_id=config.selection_id,
        selected_games_path=str(output_path),
        audit_path=str(audit_path),
        selected_game_count=len(selected_records),
        season_count=len(season_counts),
        regular_season_count=game_type_counts["regular_season"],
        playoff_count=game_type_counts["playoff"],
        regulation_count=outcome_counts["regulation"],
        overtime_only_count=outcome_counts["overtime_only"],
        shootout_count=outcome_counts["shootout"],
        selection_sha256=selection_digest,
        status="complete",
    )


def result_as_dict(
    result: MultiseasonPilotResult,
) -> dict[str, Any]:
    return asdict(result)
