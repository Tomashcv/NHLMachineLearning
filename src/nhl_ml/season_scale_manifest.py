"""Build a deterministic season-scale NHL regular-season target manifest."""

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


class SeasonScaleManifestError(ValueError):
    """Raised when a season-scale target manifest is invalid."""


@dataclass(frozen=True, slots=True)
class SeasonScaleManifestResult:
    """Summary of one season-scale manifest build."""

    corpus_id: str
    game_count: int
    season_count: int
    development_game_count: int
    validation_game_count: int
    sealed_holdout_game_count: int
    manifest_path: str
    manifest_sha256: str
    audit_path: str
    already_present: bool
    status: str


ALLOWED_SPLIT_ROLES = {
    "development",
    "validation",
    "sealed_holdout",
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise SeasonScaleManifestError(f"Invalid YAML file: {path}") from exc

    if not isinstance(payload, dict):
        raise SeasonScaleManifestError(f"Expected YAML mapping in {path}")

    return payload


def _serialize_jsonl(
    records: list[dict[str, Any]],
) -> bytes:
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


def _positive_integer(
    value: Any,
    *,
    field: str,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SeasonScaleManifestError(f"{field} must be a positive integer")

    return value


def build_season_scale_manifest(
    *,
    config_path: Path,
    manifest_path: Path,
    audit_path: Path,
) -> SeasonScaleManifestResult:
    """Build an exact candidate game-ID manifest."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Config does not exist: {config_path}")

    config_data = config_path.read_bytes()
    config_sha256 = _sha256(config_data)
    config = _read_yaml(config_path)

    corpus = config.get("corpus")
    seasons = config.get("seasons")

    if not isinstance(corpus, dict):
        raise SeasonScaleManifestError("Config has no corpus mapping")

    if not isinstance(seasons, list) or not seasons:
        raise SeasonScaleManifestError("Config seasons must be a non-empty list")

    corpus_id = str(corpus.get("corpus_id", "")).strip()
    game_type_code = str(corpus.get("game_type_code", "")).strip()
    expected_total = _positive_integer(
        corpus.get("expected_game_count"),
        field="corpus.expected_game_count",
    )

    if not corpus_id:
        raise SeasonScaleManifestError("corpus_id is missing")

    if game_type_code != "02":
        raise SeasonScaleManifestError("F5A supports regular-season code 02 only")

    if corpus.get("playoffs_included") is not False:
        raise SeasonScaleManifestError("F5A must exclude playoffs")

    manifest_rows: list[dict[str, Any]] = []
    seen_season_ids: set[str] = set()
    season_summaries: list[dict[str, Any]] = []

    for index, season in enumerate(seasons):
        if not isinstance(season, dict):
            raise SeasonScaleManifestError(f"seasons[{index}] is not a mapping")

        season_id = str(season.get("season_id", "")).strip()
        start_year = _positive_integer(
            season.get("start_year"),
            field=f"seasons[{index}].start_year",
        )
        first_sequence = _positive_integer(
            season.get("first_sequence"),
            field=(f"seasons[{index}].first_sequence"),
        )
        last_sequence = _positive_integer(
            season.get("last_sequence"),
            field=(f"seasons[{index}].last_sequence"),
        )
        expected_season_count = _positive_integer(
            season.get("expected_game_count"),
            field=(f"seasons[{index}].expected_game_count"),
        )
        split_role = str(season.get("split_role", "")).strip()

        expected_season_id = f"{start_year}{start_year + 1}"

        if season_id != expected_season_id:
            raise SeasonScaleManifestError(
                f"Season {season_id!r} does not match start year {start_year}"
            )

        if season_id in seen_season_ids:
            raise SeasonScaleManifestError(f"Duplicate season ID: {season_id}")

        if split_role not in ALLOWED_SPLIT_ROLES:
            raise SeasonScaleManifestError(f"Invalid split role for {season_id}: {split_role!r}")

        if last_sequence < first_sequence:
            raise SeasonScaleManifestError(f"Invalid sequence range for {season_id}")

        range_count = last_sequence - first_sequence + 1

        if range_count != expected_season_count:
            raise SeasonScaleManifestError(
                f"Season {season_id} range contains "
                f"{range_count} IDs, expected "
                f"{expected_season_count}"
            )

        for sequence_number in range(
            first_sequence,
            last_sequence + 1,
        ):
            game_id = f"{start_year}{game_type_code}{sequence_number:04d}"

            manifest_rows.append(
                {
                    "schema_version": "1.0",
                    "corpus_id": corpus_id,
                    "target_status": ("candidate_unverified"),
                    "requires_payload_verification": (True),
                    "season_id": season_id,
                    "start_year": start_year,
                    "game_type": "regular_season",
                    "game_type_code": game_type_code,
                    "sequence_number": (sequence_number),
                    "split_role": split_role,
                    "game_id": game_id,
                    "source_filename": (f"nhl_pbp_{game_id}.json"),
                    "source_endpoint_path": (f"/v1/gamecenter/{game_id}/play-by-play"),
                }
            )

        season_summaries.append(
            {
                "season_id": season_id,
                "start_year": start_year,
                "split_role": split_role,
                "first_sequence": first_sequence,
                "last_sequence": last_sequence,
                "game_count": range_count,
            }
        )
        seen_season_ids.add(season_id)

    manifest_rows.sort(
        key=lambda row: (
            row["start_year"],
            row["sequence_number"],
        )
    )

    game_ids = [row["game_id"] for row in manifest_rows]
    filenames = [row["source_filename"] for row in manifest_rows]
    split_counts = Counter(row["split_role"] for row in manifest_rows)
    season_counts = Counter(row["season_id"] for row in manifest_rows)

    gates = {
        "passes_expected_total_count": (len(manifest_rows) == expected_total),
        "passes_unique_game_ids": (len(set(game_ids)) == len(game_ids)),
        "passes_unique_filenames": (len(set(filenames)) == len(filenames)),
        "passes_expected_season_count": (len(seen_season_ids) == len(seasons)),
        "passes_each_season_count": all(
            season_counts[summary["season_id"]] == summary["game_count"]
            for summary in season_summaries
        ),
        "passes_game_id_length": all(
            len(game_id) == 10 and game_id.isdigit() for game_id in game_ids
        ),
        "passes_regular_season_code": all(game_id[4:6] == "02" for game_id in game_ids),
    }

    status = "complete" if all(gates.values()) else "failed"

    manifest_data = _serialize_jsonl(manifest_rows)
    manifest_sha256 = _sha256(manifest_data)
    already_present = manifest_path.is_file()

    if already_present:
        if manifest_path.read_bytes() != manifest_data:
            raise SeasonScaleManifestError(f"Existing manifest differs: {manifest_path}")
    else:
        _write_bytes_atomically(
            manifest_path,
            manifest_data,
        )

    audit = {
        "schema_version": "1.0",
        "corpus_id": corpus_id,
        "status": status,
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "game_count": len(manifest_rows),
        "season_count": len(seen_season_ids),
        "game_type": "regular_season",
        "game_type_code": game_type_code,
        "target_status": "candidate_unverified",
        "split_counts": dict(sorted(split_counts.items())),
        "season_counts": dict(sorted(season_counts.items())),
        "seasons": season_summaries,
        "gates": gates,
    }

    _write_json_atomically(audit_path, audit)

    if status != "complete":
        failed_gates = sorted(gate for gate, passed in gates.items() if not passed)
        raise SeasonScaleManifestError(f"Season-scale manifest gates failed: {failed_gates}")

    return SeasonScaleManifestResult(
        corpus_id=corpus_id,
        game_count=len(manifest_rows),
        season_count=len(seen_season_ids),
        development_game_count=(split_counts["development"]),
        validation_game_count=(split_counts["validation"]),
        sealed_holdout_game_count=(split_counts["sealed_holdout"]),
        manifest_path=str(manifest_path),
        manifest_sha256=manifest_sha256,
        audit_path=str(audit_path),
        already_present=already_present,
        status=status,
    )


def result_as_dict(
    result: SeasonScaleManifestResult,
) -> dict[str, Any]:
    """Convert a manifest result to a mapping."""
    return asdict(result)
