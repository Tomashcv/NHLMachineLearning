"""Build a frozen PBP manifest from the selected multiseason pilot."""

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


class PbpManifestError(ValueError):
    """Raised when the PBP manifest cannot be built safely."""


@dataclass(frozen=True, slots=True)
class PbpManifestResult:
    """Summary of one frozen PBP manifest build."""

    batch_id: str
    source_selection_id: str
    source_selection_sha256: str
    manifest_path: str
    manifest_sha256: str
    game_count: int
    season_count: int
    regulation_count: int
    overtime_only_count: int
    shootout_count: int
    regular_season_count: int
    playoff_count: int
    already_present: bool


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PbpManifestError(f"Invalid JSON file: {path}") from exc

    if not isinstance(payload, dict):
        raise PbpManifestError(f"Expected JSON object in {path}")

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
                raise PbpManifestError(f"Invalid JSONL in {path} at line {line_number}") from exc

            if not isinstance(record, dict):
                raise PbpManifestError(f"Expected JSON object in {path} at line {line_number}")

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


def _expected_outcome(game: dict[str, Any]) -> str:
    if game.get("went_to_shootout") is True:
        return "shootout"

    if game.get("went_to_overtime") is True:
        return "overtime_only"

    return "regulation"


def build_pbp_manifest(
    *,
    pilot_path: Path,
    selection_audit_path: Path,
    output_path: Path,
    batch_id: str,
    expected_game_count: int,
) -> PbpManifestResult:
    """Build a deterministic PBP manifest from a frozen pilot."""
    if not pilot_path.is_file():
        raise FileNotFoundError(f"Pilot file does not exist: {pilot_path}")

    if not selection_audit_path.is_file():
        raise FileNotFoundError(f"Selection audit does not exist: {selection_audit_path}")

    if not batch_id.strip():
        raise PbpManifestError("batch_id cannot be empty")

    if expected_game_count <= 0:
        raise PbpManifestError("expected_game_count must be positive")

    pilot_data = pilot_path.read_bytes()
    pilot_digest = _sha256(pilot_data)

    selection_audit = _read_json(selection_audit_path)

    if selection_audit.get("status") != "complete":
        raise PbpManifestError("Source pilot selection audit is not complete")

    audited_digest = selection_audit.get("selection_sha256")

    if audited_digest != pilot_digest:
        raise PbpManifestError("Pilot file hash does not match the selection audit")

    source_selection_id = str(selection_audit.get("selection_id", "")).strip()

    if not source_selection_id:
        raise PbpManifestError("Selection audit has no selection_id")

    records = _read_jsonl(pilot_path)

    if len(records) != expected_game_count:
        raise PbpManifestError(
            f"Pilot contains {len(records)} games, expected {expected_game_count}"
        )

    games: list[dict[str, Any]] = []
    seen_game_ids: set[str] = set()

    season_counts: Counter[str] = Counter()
    outcome_counts: Counter[str] = Counter()
    game_type_counts: Counter[str] = Counter()

    for index, record in enumerate(records):
        game = record.get("game")

        if not isinstance(game, dict):
            raise PbpManifestError(f"Record {index} is missing a game object")

        game_id = str(game.get("game_id", "")).strip()
        season_id = str(game.get("season_id", "")).strip()
        game_type = str(game.get("game_type", "")).strip()
        scheduled_start_utc = str(game.get("scheduled_start_utc", "")).strip()

        if not game_id:
            raise PbpManifestError(f"Record {index} is missing game_id")

        if game_id in seen_game_ids:
            raise PbpManifestError(f"Duplicate game ID in pilot: {game_id}")

        if game.get("status") != "final":
            raise PbpManifestError(f"Pilot game is not final: {game_id}")

        if not season_id or not game_type or not scheduled_start_utc:
            raise PbpManifestError(f"Pilot game has incomplete metadata: {game_id}")

        pilot_selection = record.get("pilot_selection")

        if not isinstance(pilot_selection, dict):
            raise PbpManifestError(f"Pilot selection metadata missing: {game_id}")

        if str(pilot_selection.get("selection_id", "")).strip() != source_selection_id:
            raise PbpManifestError(f"Selection ID mismatch for game {game_id}")

        outcome = _expected_outcome(game)

        games.append(
            {
                "game_id": game_id,
                "season_id": season_id,
                "game_type": game_type,
                "scheduled_start_utc": scheduled_start_utc,
                "expected_outcome": outcome,
                "source_filename": f"nhl_pbp_{game_id}.json",
            }
        )

        seen_game_ids.add(game_id)
        season_counts[season_id] += 1
        outcome_counts[outcome] += 1
        game_type_counts[game_type] += 1

    games.sort(
        key=lambda item: (
            item["scheduled_start_utc"],
            item["game_id"],
        )
    )

    audited_game_ids = {
        str(game_id)
        for game_id in selection_audit.get(
            "selected_game_ids",
            [],
        )
    }

    if audited_game_ids != seen_game_ids:
        raise PbpManifestError("Pilot game IDs do not match selection audit IDs")

    payload = {
        "schema_version": "1.0",
        "batch": {
            "batch_id": batch_id,
            "expected_game_count": expected_game_count,
            "source_selection_id": source_selection_id,
            "source_selection_sha256": pilot_digest,
        },
        "games": games,
    }

    manifest_text = yaml.safe_dump(
        payload,
        allow_unicode=True,
        sort_keys=False,
    )
    manifest_data = manifest_text.encode("utf-8")
    manifest_digest = _sha256(manifest_data)

    already_present = output_path.is_file()

    if already_present:
        if output_path.read_bytes() != manifest_data:
            raise PbpManifestError(f"Existing manifest differs: {output_path}")
    else:
        _write_bytes_atomically(output_path, manifest_data)

    return PbpManifestResult(
        batch_id=batch_id,
        source_selection_id=source_selection_id,
        source_selection_sha256=pilot_digest,
        manifest_path=str(output_path),
        manifest_sha256=manifest_digest,
        game_count=len(games),
        season_count=len(season_counts),
        regulation_count=outcome_counts["regulation"],
        overtime_only_count=outcome_counts["overtime_only"],
        shootout_count=outcome_counts["shootout"],
        regular_season_count=game_type_counts["regular_season"],
        playoff_count=game_type_counts["playoff"],
        already_present=already_present,
    )


def result_as_dict(
    result: PbpManifestResult,
) -> dict[str, Any]:
    """Convert a PBP manifest result into a mapping."""
    return asdict(result)
