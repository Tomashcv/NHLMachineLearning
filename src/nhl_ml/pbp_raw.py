"""Offline import and structural audit of NHL play-by-play JSON."""

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

MAX_PBP_FILE_BYTES = 50_000_000


class PbpRawError(ValueError):
    """Raised when a play-by-play raw file is invalid."""


@dataclass(frozen=True, slots=True)
class PbpRawResult:
    """Summary of one play-by-play raw import and audit."""

    game_id: str
    raw_relative_path: str
    raw_sha256: str
    byte_size: int
    play_count: int
    audit_path: str
    already_present: bool
    manifest_recorded: bool


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_json_mapping(data: bytes, path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PbpRawError(f"File is not valid UTF-8 JSON: {path}") from exc

    if not isinstance(payload, dict):
        raise PbpRawError("Expected the top-level JSON value to be an object")

    return payload


def _validate_payload(
    payload: dict[str, Any],
    expected_game_id: str,
) -> list[dict[str, Any]]:
    actual_game_id = str(payload.get("id", "")).strip()

    if not actual_game_id:
        raise PbpRawError("Play-by-play payload is missing top-level id")

    if actual_game_id != expected_game_id:
        raise PbpRawError(
            "Play-by-play game ID does not match expected game ID: "
            f"{actual_game_id!r} != {expected_game_id!r}"
        )

    plays = payload.get("plays")

    if not isinstance(plays, list):
        raise PbpRawError("Play-by-play payload must contain a plays list")

    validated: list[dict[str, Any]] = []

    for index, play in enumerate(plays):
        if not isinstance(play, dict):
            raise PbpRawError(f"plays[{index}] must be a JSON object")

        validated.append(play)

    for team_field in ("homeTeam", "awayTeam"):
        team = payload.get(team_field)

        if not isinstance(team, dict):
            raise PbpRawError(f"{team_field} must be a JSON object")

        if team.get("id") is None:
            raise PbpRawError(f"{team_field}.id is required")

    start_time = payload.get("startTimeUTC")

    if not isinstance(start_time, str) or not start_time.strip():
        raise PbpRawError("startTimeUTC is required")

    return validated


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


def _manifest_contains_record(
    manifest_path: Path,
    raw_sha256: str,
) -> bool:
    if not manifest_path.is_file():
        return False

    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()

            if not stripped:
                continue

            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise PbpRawError(
                    f"Invalid JSONL in {manifest_path} at line {line_number}"
                ) from exc

            if (
                isinstance(record, dict)
                and record.get("endpoint_family") == "play_by_play"
                and record.get("sha256") == raw_sha256
            ):
                return True

    return False


def _append_manifest_record(
    manifest_path: Path,
    record: dict[str, Any],
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    serialized = json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(serialized)
        handle.write("\n")


def _team_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {
            "id": None,
            "abbrev": None,
            "score": None,
            "shots_on_goal": None,
        }

    return {
        "id": value.get("id"),
        "abbrev": value.get("abbrev"),
        "score": value.get("score"),
        "shots_on_goal": value.get("sog"),
    }


def _build_audit(
    *,
    payload: dict[str, Any],
    plays: list[dict[str, Any]],
    raw_relative_path: str,
    raw_sha256: str,
    byte_size: int,
) -> dict[str, Any]:
    event_type_counts: Counter[str] = Counter()
    period_type_counts: Counter[str] = Counter()
    period_number_counts: Counter[str] = Counter()
    detail_key_counts: Counter[str] = Counter()

    event_ids: list[str] = []
    sort_orders: list[int] = []

    missing_event_id_count = 0
    missing_sort_order_count = 0
    plays_with_details = 0
    plays_with_coordinates = 0

    for play in plays:
        event_type = play.get("typeDescKey")
        event_type_counts[str(event_type) if event_type is not None else "<missing>"] += 1

        period = play.get("periodDescriptor")

        if isinstance(period, dict):
            period_type = period.get("periodType")
            period_number = period.get("number")

            period_type_counts[str(period_type) if period_type is not None else "<missing>"] += 1
            period_number_counts[
                str(period_number) if period_number is not None else "<missing>"
            ] += 1
        else:
            period_type_counts["<missing>"] += 1
            period_number_counts["<missing>"] += 1

        event_id = play.get("eventId")

        if event_id is None:
            missing_event_id_count += 1
        else:
            event_ids.append(str(event_id))

        sort_order = play.get("sortOrder")

        if isinstance(sort_order, int):
            sort_orders.append(sort_order)
        else:
            missing_sort_order_count += 1

        details = play.get("details")

        if isinstance(details, dict):
            plays_with_details += 1
            detail_key_counts.update(str(key) for key in details)

            if isinstance(details.get("xCoord"), int | float) and isinstance(
                details.get("yCoord"), int | float
            ):
                plays_with_coordinates += 1

    duplicate_event_ids = sorted(
        event_id for event_id, count in Counter(event_ids).items() if count > 1
    )

    sort_order_complete = missing_sort_order_count == 0
    sort_order_nondecreasing = sort_order_complete and all(
        current <= following
        for current, following in zip(
            sort_orders,
            sort_orders[1:],
            strict=False,
        )
    )

    return {
        "schema_version": "1.0",
        "source_id": "nhl_web",
        "endpoint_family": "play_by_play",
        "game_id": str(payload["id"]),
        "raw_relative_path": raw_relative_path,
        "raw_sha256": raw_sha256,
        "byte_size": byte_size,
        "game_state": payload.get("gameState"),
        "game_date": payload.get("gameDate"),
        "start_time_utc": payload.get("startTimeUTC"),
        "season_id": str(payload.get("season", "")),
        "game_type_id": payload.get("gameType"),
        "home_team": _team_summary(payload.get("homeTeam")),
        "away_team": _team_summary(payload.get("awayTeam")),
        "top_level_keys": sorted(payload),
        "play_count": len(plays),
        "event_type_counts": dict(sorted(event_type_counts.items())),
        "period_type_counts": dict(sorted(period_type_counts.items())),
        "period_number_counts": dict(sorted(period_number_counts.items())),
        "detail_key_counts": dict(sorted(detail_key_counts.items())),
        "plays_with_details": plays_with_details,
        "plays_with_coordinates": plays_with_coordinates,
        "missing_event_id_count": missing_event_id_count,
        "duplicate_event_ids": duplicate_event_ids,
        "missing_sort_order_count": missing_sort_order_count,
        "sort_order_complete": sort_order_complete,
        "sort_order_nondecreasing": sort_order_nondecreasing,
    }


def process_manual_pbp_file(
    *,
    input_path: Path,
    expected_game_id: str,
    raw_root: Path,
    manifest_path: Path,
    audit_path: Path,
) -> PbpRawResult:
    """Import and structurally audit one manually saved PBP file."""
    input_path = input_path.expanduser().resolve()
    expected_game_id = expected_game_id.strip()

    if not expected_game_id:
        raise PbpRawError("expected_game_id cannot be empty")

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    data = input_path.read_bytes()
    byte_size = len(data)

    if byte_size == 0:
        raise PbpRawError("Input file is empty")

    if byte_size > MAX_PBP_FILE_BYTES:
        raise PbpRawError(f"Input file exceeds {MAX_PBP_FILE_BYTES} bytes")

    payload = _load_json_mapping(data, input_path)
    plays = _validate_payload(payload, expected_game_id)

    digest = _sha256(data)
    imported_at_utc = datetime.now(UTC).isoformat()

    relative_path = (
        Path("nhl_web") / "manual" / "play_by_play" / expected_game_id / f"pbp_{digest[:12]}.json"
    )
    destination = raw_root / relative_path

    already_present = destination.is_file()

    if already_present:
        existing_digest = _sha256(destination.read_bytes())

        if existing_digest != digest:
            raise PbpRawError(f"Existing raw file has unexpected hash: {destination}")
    else:
        _write_bytes_atomically(destination, data)

    manifest_recorded = _manifest_contains_record(
        manifest_path,
        digest,
    )

    if not manifest_recorded:
        _append_manifest_record(
            manifest_path,
            {
                "schema_version": "1.0",
                "source_id": "nhl_web",
                "endpoint_family": "play_by_play",
                "import_mode": "manual_file",
                "game_id": expected_game_id,
                "source_filename": input_path.name,
                "raw_relative_path": relative_path.as_posix(),
                "sha256": digest,
                "byte_size": byte_size,
                "play_count": len(plays),
                "top_level_keys": sorted(payload),
                "imported_at_utc": imported_at_utc,
            },
        )
        manifest_recorded = True

    audit = _build_audit(
        payload=payload,
        plays=plays,
        raw_relative_path=relative_path.as_posix(),
        raw_sha256=digest,
        byte_size=byte_size,
    )

    audit_data = (
        json.dumps(
            audit,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")

    _write_bytes_atomically(audit_path, audit_data)

    return PbpRawResult(
        game_id=expected_game_id,
        raw_relative_path=relative_path.as_posix(),
        raw_sha256=digest,
        byte_size=byte_size,
        play_count=len(plays),
        audit_path=str(audit_path),
        already_present=already_present,
        manifest_recorded=manifest_recorded,
    )


def result_as_dict(result: PbpRawResult) -> dict[str, Any]:
    """Convert the result to a JSON-serializable mapping."""
    return asdict(result)
