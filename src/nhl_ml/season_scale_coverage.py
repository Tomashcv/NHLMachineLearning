"""Audit local PBP coverage for a season-scale target manifest."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


class SeasonScaleCoverageError(ValueError):
    """Raised when season-scale coverage is invalid."""


@dataclass(frozen=True, slots=True)
class SeasonScaleCoverageResult:
    """Summary of local season-scale PBP coverage."""

    corpus_id: str
    expected_game_count: int
    valid_game_count: int
    missing_game_count: int
    invalid_file_count: int
    unexpected_file_count: int
    duplicate_game_count: int
    coverage_fraction: float
    audit_path: str
    missing_ids_path: str
    status: str


FILE_PATTERN = re.compile(r"^nhl_pbp_(\d{10})\.json$")
FINAL_STATES = {
    "FINAL",
    "OFF",
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SeasonScaleCoverageError(f"Invalid JSON file: {path}") from exc

    if not isinstance(payload, dict):
        raise SeasonScaleCoverageError(f"Expected JSON object in {path}")

    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            stripped = line.strip()

            if not stripped:
                continue

            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise SeasonScaleCoverageError(
                    f"Invalid JSONL in {path} at line {line_number}"
                ) from exc

            if not isinstance(record, dict):
                raise SeasonScaleCoverageError(f"Expected object in {path} at line {line_number}")

            records.append(record)

    return records


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


def _discover_files(
    directories: Iterable[Path],
) -> tuple[
    dict[str, list[Path]],
    list[str],
]:
    files_by_game_id: dict[
        str,
        list[Path],
    ] = defaultdict(list)
    malformed_files: list[str] = []
    seen_paths: set[str] = set()

    for directory in directories:
        if not directory.exists():
            continue

        if not directory.is_dir():
            raise SeasonScaleCoverageError(f"Not a directory: {directory}")

        for path in directory.rglob("nhl_pbp_*.json"):
            resolved = str(path.resolve())

            if resolved in seen_paths:
                continue

            seen_paths.add(resolved)
            match = FILE_PATTERN.fullmatch(path.name)

            if match is None:
                malformed_files.append(resolved)
                continue

            files_by_game_id[match.group(1)].append(path)

    return files_by_game_id, malformed_files


def _validate_payload(
    *,
    path: Path,
    expected: dict[str, Any],
) -> tuple[bool, list[str], dict[str, Any]]:
    reasons: list[str] = []

    try:
        data = path.read_bytes()
        payload = json.loads(data)
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        return (
            False,
            [f"invalid_json:{type(exc).__name__}"],
            {},
        )

    if not isinstance(payload, dict):
        return False, ["top_level_not_object"], {}

    expected_game_id = str(expected["game_id"])
    expected_season_id = str(expected["season_id"])

    actual_game_id = _identifier(payload.get("id"))

    if actual_game_id != expected_game_id:
        reasons.append(f"game_id:{actual_game_id!r}")

    actual_season = _identifier(payload.get("season"))

    if actual_season != expected_season_id:
        reasons.append(f"season:{actual_season!r}")

    actual_game_type = _identifier(payload.get("gameType"))

    if actual_game_type not in {"2", "02"}:
        reasons.append(f"game_type:{actual_game_type!r}")

    game_state = _identifier(payload.get("gameState"))

    if game_state not in FINAL_STATES:
        reasons.append(f"game_state:{game_state!r}")

    start_time_utc = _identifier(payload.get("startTimeUTC"))

    if start_time_utc is None:
        reasons.append("missing_start_time_utc")

    away_team = payload.get("awayTeam")
    home_team = payload.get("homeTeam")

    if not isinstance(away_team, dict):
        reasons.append("missing_away_team")
        away_team_id = None
    else:
        away_team_id = _identifier(away_team.get("id"))

        if away_team_id is None:
            reasons.append("missing_away_team_id")

    if not isinstance(home_team, dict):
        reasons.append("missing_home_team")
        home_team_id = None
    else:
        home_team_id = _identifier(home_team.get("id"))

        if home_team_id is None:
            reasons.append("missing_home_team_id")

    if away_team_id is not None and home_team_id is not None and away_team_id == home_team_id:
        reasons.append("identical_team_ids")

    plays = payload.get("plays")

    if not isinstance(plays, list) or not plays:
        reasons.append("missing_or_empty_plays")

    metadata = {
        "game_id": actual_game_id,
        "season_id": actual_season,
        "game_type": actual_game_type,
        "game_state": game_state,
        "start_time_utc": start_time_utc,
        "away_team_id": away_team_id,
        "home_team_id": home_team_id,
        "play_count": (len(plays) if isinstance(plays, list) else None),
        "byte_size": len(data),
        "sha256": _sha256(data),
    }

    return not reasons, reasons, metadata


def audit_season_scale_coverage(
    *,
    manifest_path: Path,
    manifest_audit_path: Path,
    downloads_dirs: list[Path],
    audit_path: Path,
    missing_ids_path: Path,
    require_complete: bool = False,
) -> SeasonScaleCoverageResult:
    """Audit local files against an exact target manifest."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")

    if not manifest_audit_path.is_file():
        raise FileNotFoundError(f"Manifest audit does not exist: {manifest_audit_path}")

    manifest_data = manifest_path.read_bytes()
    manifest_sha256 = _sha256(manifest_data)
    manifest_audit = _read_json(manifest_audit_path)

    if manifest_audit.get("status") != "complete":
        raise SeasonScaleCoverageError("Manifest audit is not complete")

    if (
        str(
            manifest_audit.get(
                "manifest_sha256",
                "",
            )
        )
        != manifest_sha256
    ):
        raise SeasonScaleCoverageError("Manifest hash does not match audit")

    manifest_rows = _read_jsonl(manifest_path)

    expected_by_game_id: dict[
        str,
        dict[str, Any],
    ] = {}

    for row in manifest_rows:
        game_id = str(row.get("game_id", "")).strip()

        if not game_id:
            raise SeasonScaleCoverageError("Manifest row has no game_id")

        if game_id in expected_by_game_id:
            raise SeasonScaleCoverageError(f"Duplicate manifest game ID: {game_id}")

        expected_by_game_id[game_id] = row

    files_by_game_id, malformed_files = _discover_files(downloads_dirs)

    expected_ids = set(expected_by_game_id)
    discovered_ids = set(files_by_game_id)

    unexpected_ids = sorted(discovered_ids - expected_ids)
    duplicate_ids = sorted(
        game_id
        for game_id, paths in (files_by_game_id.items())
        if len(paths) > 1 and game_id in expected_ids
    )

    invalid_files: list[dict[str, Any]] = []
    valid_game_ids: list[str] = []
    missing_game_ids: list[str] = []
    valid_metadata: list[dict[str, Any]] = []

    by_season: dict[
        str,
        Counter[str],
    ] = defaultdict(Counter)
    by_split_role: dict[
        str,
        Counter[str],
    ] = defaultdict(Counter)

    for game_id, expected in expected_by_game_id.items():
        season_id = str(expected["season_id"])
        split_role = str(expected["split_role"])

        by_season[season_id]["expected"] += 1
        by_split_role[split_role]["expected"] += 1

        paths = files_by_game_id.get(
            game_id,
            [],
        )

        if not paths:
            missing_game_ids.append(game_id)
            by_season[season_id]["missing"] += 1
            by_split_role[split_role]["missing"] += 1
            continue

        if len(paths) > 1:
            by_season[season_id]["invalid"] += 1
            by_split_role[split_role]["invalid"] += 1
            continue

        path = paths[0]
        valid, reasons, metadata = _validate_payload(
            path=path,
            expected=expected,
        )

        if not valid:
            invalid_files.append(
                {
                    "game_id": game_id,
                    "season_id": season_id,
                    "split_role": split_role,
                    "path": str(path),
                    "reasons": reasons,
                }
            )
            by_season[season_id]["invalid"] += 1
            by_split_role[split_role]["invalid"] += 1
            continue

        valid_game_ids.append(game_id)
        valid_metadata.append(
            {
                **metadata,
                "path": str(path),
                "split_role": split_role,
            }
        )
        by_season[season_id]["valid"] += 1
        by_split_role[split_role]["valid"] += 1

    missing_game_ids.sort()
    valid_game_ids.sort()

    missing_lines = [
        (
            f"{game_id} "
            f"{expected_by_game_id[game_id]['season_id']} "
            f"{expected_by_game_id[game_id]['split_role']} "
            f"{expected_by_game_id[game_id]['source_filename']}"
        )
        for game_id in missing_game_ids
    ]

    missing_data = ("\n".join(missing_lines) + ("\n" if missing_lines else "")).encode("utf-8")

    _write_bytes_atomically(
        missing_ids_path,
        missing_data,
    )
    missing_ids_sha256 = _sha256(missing_data)

    expected_count = len(expected_by_game_id)
    valid_count = len(valid_game_ids)
    missing_count = len(missing_game_ids)
    invalid_count = len(invalid_files)

    if invalid_count or unexpected_ids or duplicate_ids or malformed_files:
        status = "invalid"
    elif missing_count:
        status = "partial"
    else:
        status = "complete"

    coverage_fraction = valid_count / expected_count if expected_count else 0.0

    gates = {
        "passes_expected_manifest_count": (
            expected_count
            == int(
                manifest_audit.get(
                    "game_count",
                    -1,
                )
            )
        ),
        "passes_no_invalid_files": (invalid_count == 0),
        "passes_no_unexpected_files": (not unexpected_ids and not malformed_files),
        "passes_no_duplicate_game_files": (not duplicate_ids),
        "passes_complete_coverage": (valid_count == expected_count and missing_count == 0),
    }

    audit = {
        "schema_version": "1.0",
        "corpus_id": str(manifest_audit["corpus_id"]),
        "status": status,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "downloads_dirs": [str(path) for path in downloads_dirs],
        "expected_game_count": expected_count,
        "valid_game_count": valid_count,
        "missing_game_count": missing_count,
        "invalid_file_count": invalid_count,
        "unexpected_file_count": (len(unexpected_ids) + len(malformed_files)),
        "duplicate_game_count": (len(duplicate_ids)),
        "coverage_fraction": (coverage_fraction),
        "missing_ids_path": str(missing_ids_path),
        "missing_ids_sha256": (missing_ids_sha256),
        "by_season": {
            season_id: dict(sorted(counts.items()))
            for season_id, counts in sorted(by_season.items())
        },
        "by_split_role": {
            split_role: dict(sorted(counts.items()))
            for split_role, counts in sorted(by_split_role.items())
        },
        "invalid_files": invalid_files,
        "unexpected_game_ids": (unexpected_ids),
        "malformed_files": sorted(malformed_files),
        "duplicate_game_ids": duplicate_ids,
        "valid_payload_metadata": (valid_metadata),
        "gates": gates,
    }

    _write_json_atomically(audit_path, audit)

    if require_complete and status != "complete":
        raise SeasonScaleCoverageError(
            f"Coverage is {status}: {valid_count}/{expected_count} valid"
        )

    return SeasonScaleCoverageResult(
        corpus_id=str(manifest_audit["corpus_id"]),
        expected_game_count=expected_count,
        valid_game_count=valid_count,
        missing_game_count=missing_count,
        invalid_file_count=invalid_count,
        unexpected_file_count=(len(unexpected_ids) + len(malformed_files)),
        duplicate_game_count=len(duplicate_ids),
        coverage_fraction=coverage_fraction,
        audit_path=str(audit_path),
        missing_ids_path=str(missing_ids_path),
        status=status,
    )


def result_as_dict(
    result: SeasonScaleCoverageResult,
) -> dict[str, Any]:
    """Convert a coverage result to a mapping."""
    return asdict(result)
