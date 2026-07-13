"""Audit local download coverage for a frozen NHL PBP manifest."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from nhl_ml.pbp_batch import PbpBatchConfig


class PbpCoverageError(ValueError):
    """Raised when PBP download coverage is invalid or incomplete."""


@dataclass(frozen=True, slots=True)
class PbpCoverageResult:
    """Summary of one local PBP download coverage audit."""

    batch_id: str
    expected_game_count: int
    valid_game_count: int
    missing_game_count: int
    invalid_file_count: int
    unexpected_file_count: int
    coverage_fraction: float
    audit_path: str
    status: str


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json_atomically(
    destination: Path,
    payload: dict[str, Any],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    data = (
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")

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


def _inspect_expected_file(
    *,
    path: Path,
    expected_game_id: str,
) -> dict[str, Any]:
    data = path.read_bytes()

    summary: dict[str, Any] = {
        "game_id": expected_game_id,
        "filename": path.name,
        "byte_size": len(data),
        "sha256": _sha256(data),
        "valid": False,
        "errors": [],
    }

    try:
        payload = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError):
        summary["errors"].append("invalid_utf8_json")
        return summary

    if not isinstance(payload, dict):
        summary["errors"].append("top_level_not_object")
        return summary

    actual_game_id = str(payload.get("id", "")).strip()
    summary["actual_game_id"] = actual_game_id

    if actual_game_id != expected_game_id:
        summary["errors"].append("game_id_mismatch")

    plays = payload.get("plays")

    if not isinstance(plays, list):
        summary["errors"].append("plays_not_list")
        summary["play_count"] = None
    else:
        summary["play_count"] = len(plays)

        if not plays:
            summary["errors"].append("plays_empty")

    summary["game_state"] = payload.get("gameState")
    summary["season_id"] = str(payload.get("season", ""))
    summary["game_type_id"] = payload.get("gameType")
    summary["valid"] = not summary["errors"]

    return summary


def audit_pbp_download_coverage(
    *,
    config: PbpBatchConfig,
    downloads_dir: Path,
    audit_path: Path,
    require_complete: bool = False,
) -> PbpCoverageResult:
    """Audit which configured PBP files are locally available."""
    downloads_dir = downloads_dir.expanduser().resolve()
    downloads_dir.mkdir(parents=True, exist_ok=True)

    expected_by_filename = {game.source_filename: game for game in config.games}

    valid_files: list[dict[str, Any]] = []
    invalid_files: list[dict[str, Any]] = []
    missing_games: list[dict[str, str]] = []

    observed_payload_ids: Counter[str] = Counter()

    for game in config.games:
        path = downloads_dir / game.source_filename

        if not path.is_file():
            missing_games.append(
                {
                    "game_id": game.game_id,
                    "filename": game.source_filename,
                    "download_url": (
                        f"https://api-web.nhle.com/v1/gamecenter/{game.game_id}/play-by-play"
                    ),
                }
            )
            continue

        summary = _inspect_expected_file(
            path=path,
            expected_game_id=game.game_id,
        )

        actual_game_id = summary.get("actual_game_id")

        if actual_game_id:
            observed_payload_ids[str(actual_game_id)] += 1

        if summary["valid"]:
            valid_files.append(summary)
        else:
            invalid_files.append(summary)

    unexpected_files = sorted(
        path.name for path in downloads_dir.glob("*.json") if path.name not in expected_by_filename
    )

    duplicate_payload_game_ids = sorted(
        game_id for game_id, count in observed_payload_ids.items() if count > 1
    )

    gates = {
        "passes_expected_game_count": (len(config.games) == config.expected_game_count),
        "passes_all_expected_files_present": (not missing_games),
        "passes_all_expected_files_valid": (not invalid_files),
        "passes_no_unexpected_json_files": (not unexpected_files),
        "passes_no_duplicate_payload_game_ids": (not duplicate_payload_game_ids),
    }

    if invalid_files or duplicate_payload_game_ids:
        status = "invalid"
    elif missing_games or unexpected_files:
        status = "partial"
    else:
        status = "complete"

    valid_game_count = len(valid_files)
    coverage_fraction = valid_game_count / config.expected_game_count

    audit = {
        "schema_version": "1.0",
        "batch_id": config.batch_id,
        "status": status,
        "downloads_dir": str(downloads_dir),
        "expected_game_count": config.expected_game_count,
        "valid_game_count": valid_game_count,
        "missing_game_count": len(missing_games),
        "invalid_file_count": len(invalid_files),
        "unexpected_file_count": len(unexpected_files),
        "coverage_fraction": coverage_fraction,
        "gates": gates,
        "valid_files": valid_files,
        "missing_games": missing_games,
        "invalid_files": invalid_files,
        "unexpected_files": unexpected_files,
        "duplicate_payload_game_ids": (duplicate_payload_game_ids),
    }

    _write_json_atomically(audit_path, audit)

    if require_complete and status != "complete":
        raise PbpCoverageError(
            "PBP download coverage is not complete: "
            f"status={status}, "
            f"valid={valid_game_count}, "
            f"missing={len(missing_games)}, "
            f"invalid={len(invalid_files)}, "
            f"unexpected={len(unexpected_files)}"
        )

    return PbpCoverageResult(
        batch_id=config.batch_id,
        expected_game_count=config.expected_game_count,
        valid_game_count=valid_game_count,
        missing_game_count=len(missing_games),
        invalid_file_count=len(invalid_files),
        unexpected_file_count=len(unexpected_files),
        coverage_fraction=coverage_fraction,
        audit_path=str(audit_path),
        status=status,
    )


def result_as_dict(
    result: PbpCoverageResult,
) -> dict[str, Any]:
    """Convert a PBP coverage result into a mapping."""
    return asdict(result)
