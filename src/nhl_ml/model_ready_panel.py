"""Build a leakage-safe one-row-per-game NHL model-ready panel."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


class ModelReadyPanelError(ValueError):
    """Raised when the model-ready panel cannot be built safely."""


@dataclass(frozen=True, slots=True)
class ModelReadyPanelResult:
    """Summary of one model-ready panel build."""

    dataset_id: str
    game_count: int
    row_count: int
    base_feature_count: int
    feature_column_count: int
    output_path: str
    output_sha256: str
    audit_path: str
    already_present: bool
    status: str


DATASET_ID = "nhl_regular_season_2021_2025_model_ready_v1"

EXPECTED_SPLIT_BY_SEASON = {
    "20212022": "development",
    "20222023": "development",
    "20232024": "validation",
    "20242025": "sealed_holdout",
}

WINDOW_PREFIXES = (
    "season_to_date",
    "last_3",
    "last_5",
)

WINDOW_METRIC_SUFFIXES = (
    "blocked_attempts_against_per_game",
    "blocked_attempts_for_per_game",
    "faceoff_win_percentage",
    "giveaways_for_per_game",
    "goals_against_per_game",
    "goals_for_per_game",
    "hits_for_per_game",
    "penalties_for_per_game",
    "penalty_minutes_for_per_game",
    "save_percentage_proxy",
    "shooting_percentage",
    "shot_attempt_share",
    "shot_attempts_against_per_game",
    "shot_attempts_for_per_game",
    "shots_on_goal_against_per_game",
    "shots_on_goal_for_per_game",
    "takeaways_for_per_game",
    "win_rate",
)

MODEL_FEATURE_FIELDS = (
    "days_since_previous_game_start",
    "season_to_date_games",
    "last_3_games",
    "last_5_games",
    *(f"{prefix}_{suffix}" for prefix in WINDOW_PREFIXES for suffix in WINDOW_METRIC_SUFFIXES),
)

MODEL_FEATURE_COLUMNS = tuple(
    column
    for field in MODEL_FEATURE_FIELDS
    for column in (
        f"home_{field}",
        f"away_{field}",
        f"home_minus_away_{field}",
    )
)

METADATA_COLUMNS = (
    "schema_version",
    "dataset_id",
    "game_id",
    "season_id",
    "split_role",
    "scheduled_start_utc",
    "home_team_id",
    "away_team_id",
)

TARGET_COLUMNS = (
    "target_home_win",
    "target_home_final_score",
    "target_away_final_score",
    "target_outcome_type",
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ModelReadyPanelError(f"Invalid JSON file: {path}") from exc

    if not isinstance(payload, dict):
        raise ModelReadyPanelError(f"Expected JSON object in {path}")

    return payload


def _read_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
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
                raise ModelReadyPanelError(
                    f"Invalid JSONL in {path} at line {line_number}"
                ) from exc

            if not isinstance(record, dict):
                raise ModelReadyPanelError(f"Expected JSON object in {path} at line {line_number}")

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


def _parse_utc(value: Any) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ModelReadyPanelError(f"Invalid scheduled_start_utc: {value!r}")

    normalized = value.strip().replace(
        "Z",
        "+00:00",
    )

    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ModelReadyPanelError(f"Invalid datetime: {value!r}") from exc

    if parsed.tzinfo is None:
        raise ModelReadyPanelError(f"Timezone-naive datetime: {value!r}")

    return parsed


def _integer(
    row: dict[str, Any],
    field: str,
) -> int:
    value = row.get(field)

    if isinstance(value, bool) or not isinstance(
        value,
        int,
    ):
        raise ModelReadyPanelError(f"Expected integer {field}, got {value!r}")

    return value


def _numeric_or_none(
    row: dict[str, Any],
    field: str,
) -> int | float | None:
    if field not in row:
        raise ModelReadyPanelError(f"Pregame feature field is missing: {field}")

    value = row[field]

    if value is None:
        return None

    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise ModelReadyPanelError(f"Expected numeric or null {field}, got {value!r}")

    if not math.isfinite(float(value)):
        raise ModelReadyPanelError(f"Non-finite numeric value for {field}: {value!r}")

    return value


def _difference(
    home_value: int | float | None,
    away_value: int | float | None,
) -> int | float | None:
    if home_value is None or away_value is None:
        return None

    return home_value - away_value


def _verify_source(
    *,
    source_path: Path,
    audit_path: Path,
    source_name: str,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    str,
]:
    if not source_path.is_file():
        raise FileNotFoundError(f"{source_name} does not exist: {source_path}")

    if not audit_path.is_file():
        raise FileNotFoundError(f"{source_name} audit does not exist: {audit_path}")

    source_data = source_path.read_bytes()
    source_sha256 = _sha256(source_data)
    audit = _read_json(audit_path)

    if audit.get("status") != "complete":
        raise ModelReadyPanelError(f"{source_name} audit is not complete")

    if str(audit.get("output_sha256", "")) != source_sha256:
        raise ModelReadyPanelError(f"{source_name} source hash does not match audit")

    rows = _read_jsonl(source_path)

    expected_row_count = int(audit.get("row_count", 0))

    if len(rows) != expected_row_count:
        raise ModelReadyPanelError(
            f"{source_name} contains {len(rows)} rows, expected {expected_row_count}"
        )

    return rows, audit, source_sha256


def build_model_ready_panel(
    *,
    team_game_path: Path,
    team_game_audit_path: Path,
    pregame_path: Path,
    pregame_audit_path: Path,
    output_path: Path,
    audit_path: Path,
    expected_split_by_season: dict[str, str] | None = None,
) -> ModelReadyPanelResult:
    """Build one leakage-safe model-ready row per NHL game."""
    (
        team_rows,
        team_audit,
        team_sha256,
    ) = _verify_source(
        source_path=team_game_path,
        audit_path=team_game_audit_path,
        source_name="Combined team-game panel",
    )

    (
        pregame_rows,
        pregame_audit,
        pregame_sha256,
    ) = _verify_source(
        source_path=pregame_path,
        audit_path=pregame_audit_path,
        source_name="Pregame feature panel",
    )

    if (
        str(
            pregame_audit.get(
                "source_team_game_sha256",
                "",
            )
        )
        != team_sha256
    ):
        raise ModelReadyPanelError("Pregame audit does not reference the supplied team-game panel")

    expected_game_count = int(team_audit.get("game_count", 0))

    if int(pregame_audit.get("game_count", 0)) != expected_game_count:
        raise ModelReadyPanelError("Team-game and pregame game counts differ")

    if len(team_rows) != len(pregame_rows):
        raise ModelReadyPanelError("Team-game and pregame row counts differ")

    observed_seasons = {str(row.get("season_id", "")).strip() for row in team_rows}

    if not observed_seasons or "" in observed_seasons:
        raise ModelReadyPanelError("Team-game source has invalid season IDs")

    if expected_split_by_season is None:
        unknown_seasons = observed_seasons - set(EXPECTED_SPLIT_BY_SEASON)

        if unknown_seasons:
            raise ModelReadyPanelError(
                f"Unexpected seasons in team-game source: {sorted(unknown_seasons)}"
            )

        split_by_season = {
            season_id: EXPECTED_SPLIT_BY_SEASON[season_id] for season_id in sorted(observed_seasons)
        }
    else:
        split_by_season = dict(expected_split_by_season)

    if not split_by_season:
        raise ModelReadyPanelError("Expected split-by-season mapping cannot be empty")

    allowed_split_roles = set(split_by_season.values())

    team_lookup: dict[
        tuple[str, str],
        dict[str, Any],
    ] = {}
    pregame_lookup: dict[
        tuple[str, str],
        dict[str, Any],
    ] = {}

    team_rows_by_game: dict[
        str,
        dict[str, dict[str, Any]],
    ] = defaultdict(dict)

    for row in team_rows:
        if str(row.get("schema_version")) != "1.2":
            raise ModelReadyPanelError("Expected team-game schema version 1.2")

        game_id = str(row.get("game_id", "")).strip()
        team_id = str(row.get("team_id", "")).strip()
        venue_side = str(row.get("venue_side", "")).strip()

        if not game_id or not team_id or venue_side not in {"home", "away"}:
            raise ModelReadyPanelError("Invalid team-game identifiers")

        key = (game_id, team_id)

        if key in team_lookup:
            raise ModelReadyPanelError(f"Duplicate team-game key: {key}")

        if venue_side in team_rows_by_game[game_id]:
            raise ModelReadyPanelError(f"Duplicate {venue_side} row for game {game_id}")

        team_lookup[key] = row
        team_rows_by_game[game_id][venue_side] = row

    for row in pregame_rows:
        if str(row.get("schema_version")) != "1.0":
            raise ModelReadyPanelError("Expected pregame schema version 1.0")

        if str(row.get("shots_on_goal_source")) != "official_boxscore":
            raise ModelReadyPanelError("Pregame SOG source is not official_boxscore")

        if str(row.get("goals_source")) != "canonical_pbp_non_shootout":
            raise ModelReadyPanelError("Pregame goals source is unexpected")

        game_id = str(row.get("game_id", "")).strip()
        team_id = str(row.get("team_id", "")).strip()
        key = (game_id, team_id)

        if not game_id or not team_id:
            raise ModelReadyPanelError("Invalid pregame identifiers")

        if key in pregame_lookup:
            raise ModelReadyPanelError(f"Duplicate pregame key: {key}")

        pregame_lookup[key] = row

    if set(team_lookup) != set(pregame_lookup):
        raise ModelReadyPanelError("Team-game and pregame join keys differ")

    if len(team_rows_by_game) != expected_game_count:
        raise ModelReadyPanelError(
            f"Found {len(team_rows_by_game)} games, expected {expected_game_count}"
        )

    output_rows: list[dict[str, Any]] = []

    for game_id, sides in team_rows_by_game.items():
        if set(sides) != {"home", "away"}:
            raise ModelReadyPanelError(f"Game {game_id} does not have one home and one away row")

        home_team = sides["home"]
        away_team = sides["away"]

        home_team_id = str(home_team["team_id"])
        away_team_id = str(away_team["team_id"])

        if (
            str(home_team["opponent_team_id"]) != away_team_id
            or str(away_team["opponent_team_id"]) != home_team_id
        ):
            raise ModelReadyPanelError(f"Opponent linkage is asymmetric: {game_id}")

        home_feature = pregame_lookup[(game_id, home_team_id)]
        away_feature = pregame_lookup[(game_id, away_team_id)]

        metadata_fields = (
            "game_id",
            "season_id",
            "scheduled_start_utc",
            "team_id",
            "opponent_team_id",
            "venue_side",
        )

        for field in metadata_fields:
            if str(home_team.get(field)) != str(home_feature.get(field)):
                raise ModelReadyPanelError(f"Home metadata mismatch for {game_id}: {field}")

            if str(away_team.get(field)) != str(away_feature.get(field)):
                raise ModelReadyPanelError(f"Away metadata mismatch for {game_id}: {field}")

        season_id = str(home_team["season_id"])
        split_role = str(home_team["split_role"])
        scheduled_start_utc = str(home_team["scheduled_start_utc"])

        if (
            str(away_team["season_id"]) != season_id
            or str(away_team["split_role"]) != split_role
            or str(away_team["scheduled_start_utc"]) != scheduled_start_utc
        ):
            raise ModelReadyPanelError(f"Game metadata differs between sides: {game_id}")

        expected_split = split_by_season.get(season_id)

        if expected_split is None:
            raise ModelReadyPanelError(f"Unexpected season: {season_id}")

        if split_role != expected_split:
            raise ModelReadyPanelError(f"Unexpected split role for {season_id}: {split_role}")

        _parse_utc(scheduled_start_utc)

        home_score = _integer(
            home_team,
            "official_final_score",
        )
        away_score = _integer(
            away_team,
            "official_final_score",
        )

        if home_score == away_score:
            raise ModelReadyPanelError(f"Tied final score for game {game_id}")

        home_outcome = str(home_team["expected_outcome"])
        away_outcome = str(away_team["expected_outcome"])

        if home_outcome != away_outcome:
            raise ModelReadyPanelError(f"Outcome type differs between sides: {game_id}")

        if home_outcome not in {
            "regulation",
            "overtime_only",
            "shootout",
        }:
            raise ModelReadyPanelError(f"Unexpected outcome type: {home_outcome}")

        output_row: dict[str, Any] = {
            "schema_version": "1.0",
            "dataset_id": DATASET_ID,
            "game_id": game_id,
            "season_id": season_id,
            "split_role": split_role,
            "scheduled_start_utc": (scheduled_start_utc),
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
            "target_home_win": int(home_score > away_score),
            "target_home_final_score": (home_score),
            "target_away_final_score": (away_score),
            "target_outcome_type": (home_outcome),
        }

        for field in MODEL_FEATURE_FIELDS:
            home_value = _numeric_or_none(
                home_feature,
                field,
            )
            away_value = _numeric_or_none(
                away_feature,
                field,
            )

            output_row[f"home_{field}"] = home_value
            output_row[f"away_{field}"] = away_value
            output_row[f"home_minus_away_{field}"] = _difference(
                home_value,
                away_value,
            )

        output_rows.append(output_row)

    output_rows.sort(
        key=lambda row: (
            _parse_utc(row["scheduled_start_utc"]),
            row["game_id"],
        )
    )

    game_counts = Counter(row["game_id"] for row in output_rows)
    split_game_counts = Counter(row["split_role"] for row in output_rows)
    season_game_counts = Counter(row["season_id"] for row in output_rows)

    target_values_valid = all(row["target_home_win"] in {0, 1} for row in output_rows)

    feature_columns_sha256 = _sha256(
        json.dumps(
            MODEL_FEATURE_COLUMNS,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )

    forbidden_feature_tokens = (
        "target_",
        "final_score",
        "expected_outcome",
        "official_final",
        "pbp_goals",
        "pbp_shots",
    )

    no_forbidden_feature_columns = all(
        not any(token in column for token in forbidden_feature_tokens)
        for column in MODEL_FEATURE_COLUMNS
    )

    gates = {
        "passes_expected_game_count": (len(output_rows) == expected_game_count),
        "passes_one_row_per_game": all(count == 1 for count in game_counts.values()),
        "passes_unique_game_ids": (len(game_counts) == len(output_rows)),
        "passes_expected_seasons": (set(season_game_counts) == set(split_by_season)),
        "passes_expected_split_roles": (set(split_game_counts) == allowed_split_roles),
        "passes_binary_target": (target_values_valid),
        "passes_no_target_in_features": all(
            not column.startswith("target_") for column in MODEL_FEATURE_COLUMNS
        ),
        "passes_no_postgame_feature_columns": (no_forbidden_feature_columns),
        "passes_expected_feature_count": (
            len(MODEL_FEATURE_FIELDS) == 58 and len(MODEL_FEATURE_COLUMNS) == 174
        ),
    }

    status = "complete" if all(gates.values()) else "failed"

    output_data = _serialize_jsonl(output_rows)
    output_sha256 = _sha256(output_data)
    already_present = output_path.is_file()

    if already_present:
        if output_path.read_bytes() != output_data:
            raise ModelReadyPanelError(f"Existing model-ready output differs: {output_path}")
    else:
        _write_bytes_atomically(
            output_path,
            output_data,
        )

    audit = {
        "schema_version": "1.0",
        "dataset_id": DATASET_ID,
        "status": status,
        "expected_split_by_season": dict(sorted(split_by_season.items())),
        "game_count": len(output_rows),
        "row_count": len(output_rows),
        "base_feature_count": len(MODEL_FEATURE_FIELDS),
        "feature_column_count": len(MODEL_FEATURE_COLUMNS),
        "metadata_columns": list(METADATA_COLUMNS),
        "target_columns": list(TARGET_COLUMNS),
        "base_feature_fields": list(MODEL_FEATURE_FIELDS),
        "feature_columns": list(MODEL_FEATURE_COLUMNS),
        "feature_columns_sha256": (feature_columns_sha256),
        "season_game_counts": dict(sorted(season_game_counts.items())),
        "split_game_counts": dict(sorted(split_game_counts.items())),
        "sources": {
            "team_game": {
                "path": str(team_game_path),
                "audit_path": str(team_game_audit_path),
                "sha256": team_sha256,
            },
            "pregame": {
                "path": str(pregame_path),
                "audit_path": str(pregame_audit_path),
                "sha256": pregame_sha256,
            },
        },
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
        raise ModelReadyPanelError(f"Model-ready panel gates failed: {failed_gates}")

    return ModelReadyPanelResult(
        dataset_id=DATASET_ID,
        game_count=len(output_rows),
        row_count=len(output_rows),
        base_feature_count=len(MODEL_FEATURE_FIELDS),
        feature_column_count=len(MODEL_FEATURE_COLUMNS),
        output_path=str(output_path),
        output_sha256=output_sha256,
        audit_path=str(audit_path),
        already_present=already_present,
        status=status,
    )


def result_as_dict(
    result: ModelReadyPanelResult,
) -> dict[str, Any]:
    """Convert a model-ready result into a mapping."""
    return asdict(result)
