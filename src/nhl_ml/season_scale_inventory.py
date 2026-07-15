"""Build a verified season-scale NHL game inventory from local PBP payloads."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class SeasonScaleInventoryError(ValueError):
    """Raised when the verified inventory cannot be built safely."""


@dataclass(frozen=True, slots=True)
class SeasonScaleInventoryResult:
    """Summary of one verified inventory build."""

    corpus_id: str
    game_count: int
    season_count: int
    unique_team_count: int
    regulation_count: int
    overtime_only_count: int
    shootout_count: int
    total_play_count: int
    output_path: str
    output_sha256: str
    audit_path: str
    already_present: bool
    status: str


FINAL_STATES = {"FINAL", "OFF"}

OUTCOME_BY_PERIOD_TYPE = {
    "REG": "regulation",
    "OT": "overtime_only",
    "SO": "shootout",
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SeasonScaleInventoryError(f"Invalid JSON file: {path}") from exc

    if not isinstance(payload, dict):
        raise SeasonScaleInventoryError(f"Expected JSON object in {path}")

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
                raise SeasonScaleInventoryError(
                    f"Invalid JSONL in {path} at line {line_number}"
                ) from exc

            if not isinstance(record, dict):
                raise SeasonScaleInventoryError(f"Expected object in {path} at line {line_number}")

            records.append(record)

    return records


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


def _write_json_atomically(
    destination: Path,
    payload: dict[str, Any],
) -> None:
    data = (
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")

    _write_bytes_atomically(destination, data)


def _identifier(value: Any) -> str | None:
    if value is None:
        return None

    parsed = str(value).strip()
    return parsed or None


def _required_integer(
    payload: dict[str, Any],
    field: str,
    *,
    context: str,
) -> int:
    value = payload.get(field)

    if isinstance(value, bool) or not isinstance(value, int):
        raise SeasonScaleInventoryError(f"{context}: expected integer {field}, got {value!r}")

    return value


def _canonical_utc(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SeasonScaleInventoryError(f"{context}: missing startTimeUTC")

    normalized = value.strip().replace("Z", "+00:00")

    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise SeasonScaleInventoryError(f"{context}: invalid startTimeUTC {value!r}") from exc

    if parsed.tzinfo is None:
        raise SeasonScaleInventoryError(f"{context}: timezone-naive startTimeUTC")

    return parsed.astimezone(UTC).isoformat()


def _verified_record(
    *,
    manifest_row: dict[str, Any],
    source_path: Path,
) -> tuple[dict[str, Any], set[str], set[str]]:
    game_id = str(manifest_row["game_id"])
    season_id = str(manifest_row["season_id"])
    split_role = str(manifest_row["split_role"])
    context = f"game {game_id}"

    source_data = source_path.read_bytes()

    try:
        payload = json.loads(source_data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SeasonScaleInventoryError(f"{context}: invalid UTF-8 JSON") from exc

    if not isinstance(payload, dict):
        raise SeasonScaleInventoryError(f"{context}: payload is not an object")

    actual_game_id = _identifier(payload.get("id"))

    if actual_game_id != game_id:
        raise SeasonScaleInventoryError(f"{context}: payload ID {actual_game_id!r} does not match")

    actual_season_id = _identifier(payload.get("season"))

    if actual_season_id != season_id:
        raise SeasonScaleInventoryError(
            f"{context}: payload season {actual_season_id!r} does not match {season_id!r}"
        )

    game_type = _identifier(payload.get("gameType"))

    if game_type not in {"2", "02"}:
        raise SeasonScaleInventoryError(
            f"{context}: expected regular-season gameType, got {game_type!r}"
        )

    game_state = _identifier(payload.get("gameState"))

    if game_state not in FINAL_STATES:
        raise SeasonScaleInventoryError(f"{context}: game is not final: {game_state!r}")

    scheduled_start_utc = _canonical_utc(
        payload.get("startTimeUTC"),
        context=context,
    )

    away_team = payload.get("awayTeam")
    home_team = payload.get("homeTeam")

    if not isinstance(away_team, dict):
        raise SeasonScaleInventoryError(f"{context}: missing awayTeam")

    if not isinstance(home_team, dict):
        raise SeasonScaleInventoryError(f"{context}: missing homeTeam")

    away_team_id = _identifier(away_team.get("id"))
    home_team_id = _identifier(home_team.get("id"))

    if away_team_id is None or home_team_id is None:
        raise SeasonScaleInventoryError(f"{context}: missing home or away team ID")

    if away_team_id == home_team_id:
        raise SeasonScaleInventoryError(f"{context}: identical home and away team IDs")

    away_score = _required_integer(
        away_team,
        "score",
        context=context,
    )
    home_score = _required_integer(
        home_team,
        "score",
        context=context,
    )
    away_sog = _required_integer(
        away_team,
        "sog",
        context=context,
    )
    home_sog = _required_integer(
        home_team,
        "sog",
        context=context,
    )

    game_outcome = payload.get("gameOutcome")

    if not isinstance(game_outcome, dict):
        raise SeasonScaleInventoryError(f"{context}: missing gameOutcome")

    last_period_type = _identifier(game_outcome.get("lastPeriodType"))

    expected_outcome = OUTCOME_BY_PERIOD_TYPE.get(last_period_type or "")

    if expected_outcome is None:
        raise SeasonScaleInventoryError(
            f"{context}: unsupported lastPeriodType {last_period_type!r}"
        )

    period_descriptor = payload.get("periodDescriptor")
    final_period_type: str | None = None

    if isinstance(period_descriptor, dict):
        final_period_type = _identifier(period_descriptor.get("periodType"))

    if final_period_type is not None and final_period_type != last_period_type:
        raise SeasonScaleInventoryError(
            f"{context}: periodDescriptor {final_period_type!r} "
            f"does not match gameOutcome {last_period_type!r}"
        )

    plays = payload.get("plays")

    if not isinstance(plays, list) or not plays:
        raise SeasonScaleInventoryError(f"{context}: missing or empty plays")

    roster_spots = payload.get("rosterSpots")

    if not isinstance(roster_spots, list) or not roster_spots:
        raise SeasonScaleInventoryError(f"{context}: missing or empty rosterSpots")

    top_level_keys = set(payload)
    roster_team_ids = {
        team_id
        for roster_spot in roster_spots
        if isinstance(roster_spot, dict)
        for team_id in [_identifier(roster_spot.get("teamId"))]
        if team_id is not None
    }

    valid_team_ids = {away_team_id, home_team_id}

    if not roster_team_ids.issubset(valid_team_ids):
        raise SeasonScaleInventoryError(f"{context}: roster contains an unexpected team ID")

    record = {
        "schema_version": "1.0",
        "corpus_id": str(manifest_row["corpus_id"]),
        "game_id": game_id,
        "season_id": season_id,
        "split_role": split_role,
        "sequence_number": int(manifest_row["sequence_number"]),
        "game_type": "regular_season",
        "game_state": game_state,
        "scheduled_start_utc": scheduled_start_utc,
        "away_team_id": away_team_id,
        "home_team_id": home_team_id,
        "away_score": away_score,
        "home_score": home_score,
        "away_shots_on_goal": away_sog,
        "home_shots_on_goal": home_sog,
        "provider_last_period_type": last_period_type,
        "expected_outcome": expected_outcome,
        "play_count": len(plays),
        "roster_spot_count": len(roster_spots),
        "source_filename": source_path.name,
        "source_path": str(source_path),
        "source_sha256": _sha256(source_data),
        "source_byte_size": len(source_data),
    }

    return record, top_level_keys, valid_team_ids


def build_verified_season_scale_inventory(
    *,
    manifest_path: Path,
    manifest_audit_path: Path,
    downloads_dir: Path,
    output_path: Path,
    audit_path: Path,
) -> SeasonScaleInventoryResult:
    """Build a fully verified game inventory from all local PBP payloads."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")

    if not manifest_audit_path.is_file():
        raise FileNotFoundError(f"Manifest audit does not exist: {manifest_audit_path}")

    if not downloads_dir.is_dir():
        raise FileNotFoundError(f"Downloads directory does not exist: {downloads_dir}")

    manifest_data = manifest_path.read_bytes()
    manifest_sha256 = _sha256(manifest_data)
    manifest_audit = _read_json(manifest_audit_path)

    if manifest_audit.get("status") != "complete":
        raise SeasonScaleInventoryError("Manifest audit is not complete")

    if str(manifest_audit.get("manifest_sha256", "")) != manifest_sha256:
        raise SeasonScaleInventoryError("Manifest hash does not match its audit")

    manifest_rows = _read_jsonl(manifest_path)
    expected_game_count = int(manifest_audit.get("game_count", 0))
    corpus_id = str(manifest_audit.get("corpus_id", "")).strip()

    if len(manifest_rows) != expected_game_count:
        raise SeasonScaleInventoryError(
            f"Manifest contains {len(manifest_rows)} rows, expected {expected_game_count}"
        )

    verified_rows: list[dict[str, Any]] = []
    all_team_ids: set[str] = set()
    top_level_key_counts: Counter[str] = Counter()

    for manifest_row in manifest_rows:
        game_id = str(manifest_row["game_id"])
        source_filename = str(manifest_row["source_filename"])
        source_path = downloads_dir / source_filename

        if not source_path.is_file():
            raise FileNotFoundError(f"Missing payload for game {game_id}: {source_path}")

        verified, top_level_keys, team_ids = _verified_record(
            manifest_row=manifest_row,
            source_path=source_path,
        )

        verified_rows.append(verified)
        all_team_ids.update(team_ids)
        top_level_key_counts.update(top_level_keys)

    verified_rows.sort(
        key=lambda row: (
            row["scheduled_start_utc"],
            row["game_id"],
        )
    )

    game_ids = [row["game_id"] for row in verified_rows]
    season_counts = Counter(row["season_id"] for row in verified_rows)
    split_counts = Counter(row["split_role"] for row in verified_rows)
    outcome_counts = Counter(row["expected_outcome"] for row in verified_rows)
    state_counts = Counter(row["game_state"] for row in verified_rows)

    team_start_keys = [
        (row["scheduled_start_utc"], row["away_team_id"]) for row in verified_rows
    ] + [(row["scheduled_start_utc"], row["home_team_id"]) for row in verified_rows]

    gates = {
        "passes_expected_game_count": (len(verified_rows) == expected_game_count),
        "passes_unique_game_ids": (len(set(game_ids)) == len(game_ids)),
        "passes_expected_season_counts": (
            dict(sorted(season_counts.items()))
            == dict(
                sorted(
                    manifest_audit.get(
                        "season_counts",
                        {},
                    ).items()
                )
            )
        ),
        "passes_expected_split_counts": (
            dict(sorted(split_counts.items()))
            == dict(
                sorted(
                    manifest_audit.get(
                        "split_counts",
                        {},
                    ).items()
                )
            )
        ),
        "passes_recognized_outcomes_only": (
            set(outcome_counts)
            <= {
                "regulation",
                "overtime_only",
                "shootout",
            }
            and bool(outcome_counts)
        ),
        "passes_unique_team_start_keys": (len(set(team_start_keys)) == len(team_start_keys)),
        "passes_nonempty_team_universe": (len(all_team_ids) > 0),
    }

    status = "complete" if all(gates.values()) else "failed"

    output_data = _serialize_jsonl(verified_rows)
    output_sha256 = _sha256(output_data)
    already_present = output_path.is_file()

    if already_present:
        if output_path.read_bytes() != output_data:
            raise SeasonScaleInventoryError(f"Existing verified inventory differs: {output_path}")
    else:
        _write_bytes_atomically(output_path, output_data)

    audit = {
        "schema_version": "1.0",
        "corpus_id": corpus_id,
        "status": status,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "downloads_dir": str(downloads_dir),
        "game_count": len(verified_rows),
        "season_count": len(season_counts),
        "unique_team_count": len(all_team_ids),
        "team_ids": sorted(all_team_ids, key=int),
        "total_play_count": sum(row["play_count"] for row in verified_rows),
        "total_source_byte_size": sum(row["source_byte_size"] for row in verified_rows),
        "season_counts": dict(sorted(season_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "game_state_counts": dict(sorted(state_counts.items())),
        "minimum_scheduled_start_utc": (verified_rows[0]["scheduled_start_utc"]),
        "maximum_scheduled_start_utc": (verified_rows[-1]["scheduled_start_utc"]),
        "top_level_key_counts": dict(sorted(top_level_key_counts.items())),
        "output_path": str(output_path),
        "output_sha256": output_sha256,
        "gates": gates,
    }

    _write_json_atomically(audit_path, audit)

    if status != "complete":
        failed_gates = sorted(gate for gate, passed in gates.items() if not passed)
        raise SeasonScaleInventoryError(f"Verified inventory gates failed: {failed_gates}")

    return SeasonScaleInventoryResult(
        corpus_id=corpus_id,
        game_count=len(verified_rows),
        season_count=len(season_counts),
        unique_team_count=len(all_team_ids),
        regulation_count=outcome_counts["regulation"],
        overtime_only_count=outcome_counts["overtime_only"],
        shootout_count=outcome_counts["shootout"],
        total_play_count=sum(row["play_count"] for row in verified_rows),
        output_path=str(output_path),
        output_sha256=output_sha256,
        audit_path=str(audit_path),
        already_present=already_present,
        status=status,
    )


def result_as_dict(
    result: SeasonScaleInventoryResult,
) -> dict[str, Any]:
    """Convert a verified inventory result into a mapping."""
    return asdict(result)
