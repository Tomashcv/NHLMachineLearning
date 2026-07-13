"""Offline import utilities for manually saved NHL web JSON files."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

MAX_RAW_FILE_BYTES = 25_000_000


class NHLRawImportError(ValueError):
    """Raised when a manually saved NHL file fails validation."""


@dataclass(frozen=True, slots=True)
class RawImportResult:
    """Summary of one raw-file import."""

    source_id: str
    endpoint_family: str
    requested_date: str
    source_filename: str
    raw_relative_path: str
    sha256: str
    byte_size: int
    game_count: int
    imported_at_utc: str
    already_present: bool
    manifest_recorded: bool


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_json_mapping(data: bytes, source_path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NHLRawImportError(f"File is not valid UTF-8 JSON: {source_path}") from exc

    if not isinstance(payload, dict):
        raise NHLRawImportError("Expected the top-level JSON value to be an object")

    return payload


def _validate_score_payload(
    payload: dict[str, Any],
    requested_date: date,
) -> int:
    games = payload.get("games")

    if not isinstance(games, list):
        raise NHLRawImportError("Expected the NHL score payload to contain a 'games' list")

    current_date = payload.get("currentDate")
    if current_date is not None and current_date != requested_date.isoformat():
        raise NHLRawImportError(
            "Payload currentDate does not match the requested date: "
            f"{current_date!r} != {requested_date.isoformat()!r}"
        )

    for index, game in enumerate(games):
        if not isinstance(game, dict):
            raise NHLRawImportError(f"games[{index}] must be a JSON object")

    return len(games)


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


def _manifest_contains_hash(manifest_path: Path, sha256: str) -> bool:
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
                raise NHLRawImportError(
                    f"Invalid JSONL record in {manifest_path} at line {line_number}"
                ) from exc

            if isinstance(record, dict) and record.get("sha256") == sha256:
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


def import_manual_score_file(
    *,
    input_path: Path,
    requested_date: date,
    raw_root: Path,
    manifest_path: Path,
) -> RawImportResult:
    """Import one manually saved NHL daily-score JSON file.

    This function performs no network access.
    """
    input_path = input_path.expanduser().resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    data = input_path.read_bytes()
    byte_size = len(data)

    if byte_size == 0:
        raise NHLRawImportError("Input file is empty")

    if byte_size > MAX_RAW_FILE_BYTES:
        raise NHLRawImportError(f"Input file exceeds {MAX_RAW_FILE_BYTES} bytes")

    payload = _load_json_mapping(data, input_path)
    game_count = _validate_score_payload(payload, requested_date)

    digest = _sha256(data)
    imported_at = datetime.now(UTC).isoformat()

    relative_path = (
        Path("nhl_web") / "manual" / requested_date.isoformat() / f"score_{digest[:12]}.json"
    )
    destination = raw_root / relative_path

    already_present = destination.is_file()

    if already_present:
        existing_digest = _sha256(destination.read_bytes())
        if existing_digest != digest:
            raise NHLRawImportError(f"Existing raw file has an unexpected hash: {destination}")
    else:
        _write_bytes_atomically(destination, data)

    manifest_recorded = _manifest_contains_hash(manifest_path, digest)

    record = {
        "schema_version": "1.0",
        "source_id": "nhl_web",
        "endpoint_family": "daily_score",
        "import_mode": "manual_file",
        "requested_date": requested_date.isoformat(),
        "source_filename": input_path.name,
        "raw_relative_path": relative_path.as_posix(),
        "sha256": digest,
        "byte_size": byte_size,
        "game_count": game_count,
        "top_level_keys": sorted(payload),
        "imported_at_utc": imported_at,
    }

    if not manifest_recorded:
        _append_manifest_record(manifest_path, record)
        manifest_recorded = True

    return RawImportResult(
        source_id="nhl_web",
        endpoint_family="daily_score",
        requested_date=requested_date.isoformat(),
        source_filename=input_path.name,
        raw_relative_path=relative_path.as_posix(),
        sha256=digest,
        byte_size=byte_size,
        game_count=game_count,
        imported_at_utc=imported_at,
        already_present=already_present,
        manifest_recorded=manifest_recorded,
    )


def result_as_dict(result: RawImportResult) -> dict[str, Any]:
    """Convert an import result into a JSON-serializable mapping."""
    return asdict(result)
