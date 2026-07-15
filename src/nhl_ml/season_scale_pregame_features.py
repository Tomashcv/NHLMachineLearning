"""Build the combined season-scale team-game source for pregame features."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from nhl_ml.rolling_features import RollingFeatureError


@dataclass(frozen=True, slots=True)
class SeasonScaleTeamGamePanelResult:
    """Summary of a combined season-scale team-game panel."""

    batch_id: str
    season_count: int
    game_count: int
    row_count: int
    team_season_count: int
    output_path: str
    output_sha256: str
    audit_path: str
    already_present: bool
    status: str


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RollingFeatureError(f"Invalid JSON file: {path}") from exc

    if not isinstance(payload, dict):
        raise RollingFeatureError(f"Expected JSON object in {path}")

    return payload


def _read_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
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
                raise RollingFeatureError(f"Invalid JSONL in {path} at line {line_number}") from exc

            if not isinstance(record, dict):
                raise RollingFeatureError(f"Expected JSON object in {path} at line {line_number}")

            records.append(record)

    return records


def _serialize_jsonl(
    records: list[dict[str, Any]],
) -> bytes:
    serialized = "\n".join(
        json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for record in records
    )

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

        os.replace(
            temporary_path,
            destination,
        )
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

    _write_bytes_atomically(
        destination,
        data,
    )


def build_season_scale_team_game_panel(
    *,
    season_ids: tuple[str, ...],
    team_game_root: Path,
    source_audit_root: Path,
    output_path: Path,
    audit_path: Path,
    expected_team_games_per_season: int | None = 82,
) -> SeasonScaleTeamGamePanelResult:
    """Combine audited season files into one deterministic team-game panel."""
    if not season_ids:
        raise RollingFeatureError("At least one season is required")

    all_rows: list[dict[str, Any]] = []
    source_summaries: dict[
        str,
        dict[str, Any],
    ] = {}
    corpus_ids: set[str] = set()

    expected_total_games = 0
    expected_total_rows = 0

    for season_id in season_ids:
        source_path = team_game_root / f"team_game_{season_id}.jsonl"
        source_audit_path = source_audit_root / f"team_game_{season_id}.json"

        if not source_path.is_file():
            raise FileNotFoundError(f"Missing team-game source: {source_path}")

        if not source_audit_path.is_file():
            raise FileNotFoundError(f"Missing team-game audit: {source_audit_path}")

        source_data = source_path.read_bytes()
        source_sha256 = _sha256(source_data)
        source_audit = _read_json(source_audit_path)

        if source_audit.get("status") != "complete":
            raise RollingFeatureError(f"Team-game audit is not complete: {season_id}")

        if (
            str(
                source_audit.get(
                    "output_sha256",
                    "",
                )
            )
            != source_sha256
        ):
            raise RollingFeatureError(f"Team-game source hash mismatch: {season_id}")

        if (
            str(
                source_audit.get(
                    "season_id",
                    "",
                )
            )
            != season_id
        ):
            raise RollingFeatureError(f"Team-game audit season mismatch: {season_id}")

        source_rows = _read_jsonl(source_path)
        expected_rows = int(
            source_audit.get(
                "row_count",
                0,
            )
        )
        expected_games = int(
            source_audit.get(
                "game_count",
                0,
            )
        )

        if len(source_rows) != expected_rows:
            raise RollingFeatureError(
                f"{season_id} contains {len(source_rows)} rows, expected {expected_rows}"
            )

        season_game_ids: set[str] = set()

        for row in source_rows:
            if (
                str(
                    row.get(
                        "schema_version",
                        "",
                    )
                )
                != "1.2"
            ):
                raise RollingFeatureError(f"Expected season-scale schema 1.2 in {season_id}")

            if str(row.get("season_id")) != season_id:
                raise RollingFeatureError(f"Row season mismatch in {season_id}")

            game_id = str(row.get("game_id", "")).strip()

            if not game_id:
                raise RollingFeatureError(f"Missing game ID in {season_id}")

            season_game_ids.add(game_id)

            corpus_id = str(row.get("corpus_id", "")).strip()

            if not corpus_id:
                raise RollingFeatureError(f"Missing corpus ID in {season_id}")

            corpus_ids.add(corpus_id)

        if len(season_game_ids) != expected_games:
            raise RollingFeatureError(
                f"{season_id} contains {len(season_game_ids)} games, expected {expected_games}"
            )

        expected_total_rows += expected_rows
        expected_total_games += expected_games
        all_rows.extend(source_rows)

        source_summaries[season_id] = {
            "split_role": str(source_audit["split_role"]),
            "game_count": expected_games,
            "row_count": expected_rows,
            "source_path": str(source_path),
            "source_sha256": source_sha256,
            "source_audit_path": str(source_audit_path),
        }

    if len(corpus_ids) != 1:
        raise RollingFeatureError(f"Expected one corpus ID, found {sorted(corpus_ids)}")

    corpus_id = next(iter(corpus_ids))
    batch_id = "nhl_regular_season_2021_2025_team_game_v1"

    all_rows.sort(
        key=lambda row: (
            str(row["scheduled_start_utc"]),
            str(row["game_id"]),
            0 if row["venue_side"] == "away" else 1,
        )
    )

    game_team_counts = Counter(
        (
            str(row["game_id"]),
            str(row["team_id"]),
        )
        for row in all_rows
    )
    game_row_counts = Counter(str(row["game_id"]) for row in all_rows)
    season_row_counts = Counter(str(row["season_id"]) for row in all_rows)
    split_row_counts = Counter(str(row["split_role"]) for row in all_rows)
    team_season_game_counts = Counter(
        (
            str(row["season_id"]),
            str(row["team_id"]),
        )
        for row in all_rows
    )

    expected_team_schedule_passed = (
        True
        if expected_team_games_per_season is None
        else all(
            count == expected_team_games_per_season for count in team_season_game_counts.values()
        )
    )

    gates = {
        "passes_expected_season_count": (len(season_row_counts) == len(season_ids)),
        "passes_expected_game_count": (len(game_row_counts) == expected_total_games),
        "passes_expected_row_count": (len(all_rows) == expected_total_rows),
        "passes_two_rows_per_game": all(count == 2 for count in game_row_counts.values()),
        "passes_unique_game_team_keys": all(count == 1 for count in game_team_counts.values()),
        "passes_expected_team_schedule": (expected_team_schedule_passed),
        "passes_single_corpus": (len(corpus_ids) == 1),
    }

    status = "complete" if all(gates.values()) else "failed"

    output_data = _serialize_jsonl(all_rows)
    output_sha256 = _sha256(output_data)
    already_present = output_path.is_file()

    if already_present:
        if output_path.read_bytes() != output_data:
            raise RollingFeatureError(f"Existing combined team-game panel differs: {output_path}")
    else:
        _write_bytes_atomically(
            output_path,
            output_data,
        )

    audit = {
        "schema_version": "1.0",
        "batch_id": batch_id,
        "corpus_id": corpus_id,
        "status": status,
        "season_count": len(season_row_counts),
        "game_count": len(game_row_counts),
        "row_count": len(all_rows),
        "team_season_count": len(team_season_game_counts),
        "expected_team_games_per_season": (expected_team_games_per_season),
        "season_row_counts": dict(sorted(season_row_counts.items())),
        "split_row_counts": dict(sorted(split_row_counts.items())),
        "team_season_game_count_distribution": dict(
            sorted(Counter(team_season_game_counts.values()).items())
        ),
        "sources": source_summaries,
        "gates": gates,
        "output_path": str(output_path),
        "output_sha256": output_sha256,
    }

    _write_json_atomically(
        audit_path,
        audit,
    )

    if status != "complete":
        failed_gates = sorted(gate for gate, passed in gates.items() if not passed)
        raise RollingFeatureError(f"Combined team-game panel gates failed: {failed_gates}")

    return SeasonScaleTeamGamePanelResult(
        batch_id=batch_id,
        season_count=len(season_row_counts),
        game_count=len(game_row_counts),
        row_count=len(all_rows),
        team_season_count=len(team_season_game_counts),
        output_path=str(output_path),
        output_sha256=output_sha256,
        audit_path=str(audit_path),
        already_present=already_present,
        status=status,
    )


def panel_result_as_dict(
    result: SeasonScaleTeamGamePanelResult,
) -> dict[str, Any]:
    """Convert a combined-panel result into a mapping."""
    return asdict(result)
