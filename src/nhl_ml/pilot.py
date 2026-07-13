"""Small, deterministic NHL pilot construction."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class PilotBuildError(ValueError):
    """Raised when pilot artefacts cannot be built safely."""


@dataclass(frozen=True, slots=True)
class PilotBuildResult:
    """Summary of one five-game pilot build."""

    selected_games_path: str
    teams_path: str
    audit_path: str
    selected_game_count: int
    team_count: int
    regulation_count: int
    overtime_only_count: int
    shootout_count: int
    missing_required_outcome_types: tuple[str, ...]
    status: str


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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
                raise PilotBuildError(f"Invalid JSONL in {path} at line {line_number}") from exc

            if not isinstance(record, dict):
                raise PilotBuildError(f"Expected a JSON object in {path} at line {line_number}")

            records.append(record)

    return records


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


def _discover_game_records(
    canonical_root: Path,
) -> list[dict[str, Any]]:
    files = sorted(canonical_root.glob("*/games_*.jsonl"))

    if not files:
        raise PilotBuildError(f"No canonical NHL game files found under {canonical_root}")

    records_by_game_id: dict[str, dict[str, Any]] = {}

    for path in files:
        for record in _read_jsonl(path):
            game = record.get("game")

            if not isinstance(game, dict):
                raise PilotBuildError(f"Canonical record in {path} has no game object")

            game_id = str(game.get("game_id", "")).strip()

            if not game_id:
                raise PilotBuildError(f"Canonical record in {path} has no game_id")

            existing = records_by_game_id.get(game_id)

            if existing is not None and existing != record:
                raise PilotBuildError(f"Conflicting canonical records for game {game_id}")

            records_by_game_id[game_id] = record

    return sorted(
        records_by_game_id.values(),
        key=lambda record: (
            record["game"]["scheduled_start_utc"],
            record["game"]["game_id"],
        ),
    )


def _outcome_type(record: dict[str, Any]) -> str:
    game = record["game"]

    if game.get("status") != "final":
        return "not_final"

    if game.get("went_to_shootout") is True:
        return "shootout"

    if game.get("went_to_overtime") is True:
        return "overtime_only"

    return "regulation"


def _select_games(
    records: list[dict[str, Any]],
    target_games: int,
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    final_records = [record for record in records if record["game"].get("status") == "final"]

    if len(final_records) < target_games:
        raise PilotBuildError(
            f"Only {len(final_records)} final games are available; {target_games} are required"
        )

    required_types = (
        "regulation",
        "overtime_only",
        "shootout",
    )

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    missing: list[str] = []

    for required_type in required_types:
        candidate = next(
            (record for record in final_records if _outcome_type(record) == required_type),
            None,
        )

        if candidate is None:
            missing.append(required_type)
            continue

        selected.append(candidate)
        selected_ids.add(candidate["game"]["game_id"])

    for record in final_records:
        if len(selected) >= target_games:
            break

        game_id = record["game"]["game_id"]

        if game_id in selected_ids:
            continue

        selected.append(record)
        selected_ids.add(game_id)

    selected.sort(
        key=lambda record: (
            record["game"]["scheduled_start_utc"],
            record["game"]["game_id"],
        )
    )

    return selected, tuple(missing)


def _load_raw_game(
    *,
    canonical_record: dict[str, Any],
    raw_root: Path,
) -> dict[str, Any]:
    raw_relative_path = canonical_record.get("source_raw_relative_path")
    raw_sha256 = canonical_record.get("source_raw_sha256")

    if not isinstance(raw_relative_path, str):
        raise PilotBuildError("Canonical record is missing source_raw_relative_path")

    raw_file = raw_root / raw_relative_path

    if not raw_file.is_file():
        raise PilotBuildError(f"Raw source file does not exist: {raw_file}")

    raw_data = raw_file.read_bytes()

    if _sha256(raw_data) != raw_sha256:
        raise PilotBuildError(f"Raw source hash mismatch: {raw_file}")

    try:
        payload = json.loads(raw_data)
    except json.JSONDecodeError as exc:
        raise PilotBuildError(f"Raw source is not valid JSON: {raw_file}") from exc

    game_id = canonical_record["game"]["game_id"]

    for game in payload.get("games", []):
        if str(game.get("id")) == game_id:
            return game

    raise PilotBuildError(f"Game {game_id} was not found in its linked raw file")


def _build_team_records(
    *,
    selected_games: list[dict[str, Any]],
    raw_root: Path,
) -> list[dict[str, Any]]:
    team_observations: dict[str, dict[str, Any]] = {}

    for canonical_record in selected_games:
        game = canonical_record["game"]
        raw_game = _load_raw_game(
            canonical_record=canonical_record,
            raw_root=raw_root,
        )

        for side in ("homeTeam", "awayTeam"):
            team = raw_game.get(side)

            if not isinstance(team, dict):
                raise PilotBuildError(f"{side} is missing for game {game['game_id']}")

            team_id = str(team.get("id", "")).strip()
            abbreviation = str(team.get("abbrev", "")).strip()

            if not team_id or not abbreviation:
                raise PilotBuildError(f"Missing team identity in game {game['game_id']}")

            observation = team_observations.setdefault(
                team_id,
                {
                    "schema_version": "1.0",
                    "team_id": team_id,
                    "abbreviations": set(),
                    "source_game_ids": set(),
                    "pilot_first_seen_utc": game["scheduled_start_utc"],
                    "pilot_last_seen_utc": game["scheduled_start_utc"],
                },
            )

            observation["abbreviations"].add(abbreviation)
            observation["source_game_ids"].add(game["game_id"])
            observation["pilot_first_seen_utc"] = min(
                observation["pilot_first_seen_utc"],
                game["scheduled_start_utc"],
            )
            observation["pilot_last_seen_utc"] = max(
                observation["pilot_last_seen_utc"],
                game["scheduled_start_utc"],
            )

    records: list[dict[str, Any]] = []

    for team_id in sorted(team_observations, key=int):
        observation = team_observations[team_id]

        records.append(
            {
                "schema_version": observation["schema_version"],
                "team_id": team_id,
                "abbreviations": sorted(observation["abbreviations"]),
                "source_game_ids": sorted(observation["source_game_ids"]),
                "pilot_first_seen_utc": observation["pilot_first_seen_utc"],
                "pilot_last_seen_utc": observation["pilot_last_seen_utc"],
                "identity_scope": "pilot_observation_only",
            }
        )

    return records


def build_five_game_pilot(
    *,
    canonical_root: Path,
    raw_root: Path,
    output_root: Path,
    audit_path: Path,
    target_games: int = 5,
) -> PilotBuildResult:
    """Build the first deterministic NHL pilot artefacts."""
    records = _discover_game_records(canonical_root)
    selected, missing = _select_games(records, target_games)

    team_records = _build_team_records(
        selected_games=selected,
        raw_root=raw_root,
    )

    outcome_counts = {
        "regulation": 0,
        "overtime_only": 0,
        "shootout": 0,
    }

    for record in selected:
        outcome_counts[_outcome_type(record)] += 1

    selected_path = output_root / "five_games.jsonl"
    teams_path = output_root / "teams.jsonl"

    selected_data = _serialize_jsonl(selected)
    teams_data = _serialize_jsonl(team_records)

    _write_bytes_atomically(selected_path, selected_data)
    _write_bytes_atomically(teams_path, teams_data)

    status = "complete_outcome_coverage" if not missing else "incomplete_outcome_coverage"

    audit_record = {
        "schema_version": "1.0",
        "status": status,
        "target_game_count": target_games,
        "selected_game_count": len(selected),
        "team_count": len(team_records),
        "outcome_counts": outcome_counts,
        "missing_required_outcome_types": list(missing),
        "selected_game_ids": [record["game"]["game_id"] for record in selected],
        "selected_games_sha256": _sha256(selected_data),
        "teams_sha256": _sha256(teams_data),
    }

    audit_data = (
        json.dumps(
            audit_record,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")

    _write_bytes_atomically(audit_path, audit_data)

    return PilotBuildResult(
        selected_games_path=str(selected_path),
        teams_path=str(teams_path),
        audit_path=str(audit_path),
        selected_game_count=len(selected),
        team_count=len(team_records),
        regulation_count=outcome_counts["regulation"],
        overtime_only_count=outcome_counts["overtime_only"],
        shootout_count=outcome_counts["shootout"],
        missing_required_outcome_types=missing,
        status=status,
    )


def result_as_dict(result: PilotBuildResult) -> dict[str, Any]:
    """Convert the pilot result into a serializable mapping."""
    return asdict(result)
