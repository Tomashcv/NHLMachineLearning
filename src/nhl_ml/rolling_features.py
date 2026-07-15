"""Build leakage-safe pre-game rolling team features."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


class RollingFeatureError(ValueError):
    """Raised when rolling features cannot be built safely."""


@dataclass(frozen=True, slots=True)
class RollingFeatureResult:
    """Summary of one pre-game rolling feature build."""

    source_batch_id: str
    game_count: int
    row_count: int
    rows_with_any_history: int
    rows_with_three_history_games: int
    rows_with_five_history_games: int
    output_path: str
    output_sha256: str
    audit_path: str
    already_present: bool
    status: str


HISTORY_RULE = "same_season_prior_utc_dates_only"

SUPPORTED_TEAM_GAME_SCHEMA_VERSIONS = {
    "1.1",
    "1.2",
}

SHOTS_ON_GOAL_SOURCE = "official_boxscore"
GOALS_SOURCE = "canonical_pbp_non_shootout"


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
                raise RollingFeatureError(f"Invalid JSONL in {path} at line {line_number}") from exc

            if not isinstance(record, dict):
                raise RollingFeatureError(f"Expected JSON object in {path} at line {line_number}")

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


def _parse_utc(value: Any) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise RollingFeatureError(f"Invalid scheduled_start_utc: {value!r}")

    normalized = value.strip().replace("Z", "+00:00")

    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise RollingFeatureError(f"Invalid datetime: {value!r}") from exc

    if parsed.tzinfo is None:
        raise RollingFeatureError(f"Timezone-naive datetime: {value!r}")

    return parsed


def _integer(
    row: dict[str, Any],
    field: str,
) -> int:
    value = row.get(field)

    if isinstance(value, bool) or not isinstance(value, int):
        raise RollingFeatureError(f"Expected integer {field}, got {value!r}")

    return value


def _safe_divide(
    numerator: float,
    denominator: float,
) -> float | None:
    if denominator == 0:
        return None

    return numerator / denominator


def _window_features(
    history: list[dict[str, Any]],
    *,
    prefix: str,
) -> dict[str, int | float | None]:
    game_count = len(history)

    if game_count == 0:
        return {
            f"{prefix}_games": 0,
            f"{prefix}_win_rate": None,
            f"{prefix}_goals_for_per_game": None,
            f"{prefix}_goals_against_per_game": None,
            f"{prefix}_shots_on_goal_for_per_game": None,
            f"{prefix}_shots_on_goal_against_per_game": None,
            f"{prefix}_shot_attempts_for_per_game": None,
            f"{prefix}_shot_attempts_against_per_game": None,
            f"{prefix}_blocked_attempts_for_per_game": None,
            f"{prefix}_blocked_attempts_against_per_game": None,
            f"{prefix}_shooting_percentage": None,
            f"{prefix}_save_percentage_proxy": None,
            f"{prefix}_shot_attempt_share": None,
            f"{prefix}_faceoff_win_percentage": None,
            f"{prefix}_penalties_for_per_game": None,
            f"{prefix}_penalty_minutes_for_per_game": None,
            f"{prefix}_hits_for_per_game": None,
            f"{prefix}_giveaways_for_per_game": None,
            f"{prefix}_takeaways_for_per_game": None,
        }

    sums: Counter[str] = Counter()

    for performance in history:
        for key, value in performance.items():
            if isinstance(value, int):
                sums[key] += value

    game_count_float = float(game_count)

    shots_for = sums["shots_on_goal_for"]
    shots_against = sums["shots_on_goal_against"]
    attempts_for = sums["shot_attempts_for"]
    attempts_against = sums["shot_attempts_against"]
    faceoffs_for = sums["faceoff_wins_for"]
    faceoffs_against = sums["faceoff_wins_against"]

    return {
        f"{prefix}_games": game_count,
        f"{prefix}_win_rate": (sums["win"] / game_count_float),
        f"{prefix}_goals_for_per_game": (sums["goals_for"] / game_count_float),
        f"{prefix}_goals_against_per_game": (sums["goals_against"] / game_count_float),
        f"{prefix}_shots_on_goal_for_per_game": (shots_for / game_count_float),
        f"{prefix}_shots_on_goal_against_per_game": (shots_against / game_count_float),
        f"{prefix}_shot_attempts_for_per_game": (attempts_for / game_count_float),
        f"{prefix}_shot_attempts_against_per_game": (attempts_against / game_count_float),
        f"{prefix}_blocked_attempts_for_per_game": (
            sums["blocked_attempts_for"] / game_count_float
        ),
        f"{prefix}_blocked_attempts_against_per_game": (
            sums["blocked_attempts_against"] / game_count_float
        ),
        f"{prefix}_shooting_percentage": _safe_divide(
            sums["goals_for"],
            shots_for,
        ),
        f"{prefix}_save_percentage_proxy": (
            None if shots_against == 0 else 1.0 - (sums["goals_against"] / shots_against)
        ),
        f"{prefix}_shot_attempt_share": _safe_divide(
            attempts_for,
            attempts_for + attempts_against,
        ),
        f"{prefix}_faceoff_win_percentage": _safe_divide(
            faceoffs_for,
            faceoffs_for + faceoffs_against,
        ),
        f"{prefix}_penalties_for_per_game": (sums["penalties_for"] / game_count_float),
        f"{prefix}_penalty_minutes_for_per_game": (sums["penalty_minutes_for"] / game_count_float),
        f"{prefix}_hits_for_per_game": (sums["hits_for"] / game_count_float),
        f"{prefix}_giveaways_for_per_game": (sums["giveaways_for"] / game_count_float),
        f"{prefix}_takeaways_for_per_game": (sums["takeaways_for"] / game_count_float),
    }


def build_pregame_rolling_features(
    *,
    team_game_path: Path,
    team_game_audit_path: Path,
    output_path: Path,
    audit_path: Path,
) -> RollingFeatureResult:
    """Build pre-game features using prior UTC dates only."""
    if not team_game_path.is_file():
        raise FileNotFoundError(f"Team-game file does not exist: {team_game_path}")

    if not team_game_audit_path.is_file():
        raise FileNotFoundError(f"Team-game audit does not exist: {team_game_audit_path}")

    source_data = team_game_path.read_bytes()
    source_sha256 = _sha256(source_data)
    source_audit = _read_json(team_game_audit_path)

    if source_audit.get("status") != "complete":
        raise RollingFeatureError("Team-game source audit is not complete")

    if str(source_audit.get("output_sha256", "")) != source_sha256:
        raise RollingFeatureError("Team-game source hash does not match audit")

    source_batch_id = str(source_audit.get("batch_id", "")).strip()

    expected_row_count = int(source_audit.get("row_count", 0))
    expected_game_count = int(source_audit.get("game_count", 0))

    if not source_batch_id:
        raise RollingFeatureError("Team-game source batch ID is missing")

    source_rows = _read_jsonl(team_game_path)

    if len(source_rows) != expected_row_count:
        raise RollingFeatureError(
            f"Source contains {len(source_rows)} rows, expected {expected_row_count}"
        )

    rows_by_game: dict[
        str,
        dict[str, dict[str, Any]],
    ] = defaultdict(dict)

    seen_keys: set[tuple[str, str]] = set()

    for row in source_rows:
        schema_version = str(row.get("schema_version", ""))

        if schema_version not in SUPPORTED_TEAM_GAME_SCHEMA_VERSIONS:
            raise RollingFeatureError(f"Unsupported team-game schema version: {schema_version!r}")

        game_id = str(row.get("game_id", "")).strip()
        team_id = str(row.get("team_id", "")).strip()

        if not game_id or not team_id:
            raise RollingFeatureError("Team-game row has missing identifiers")

        key = (game_id, team_id)

        if key in seen_keys:
            raise RollingFeatureError(f"Duplicate game-team key: {key}")

        rows_by_game[game_id][team_id] = row
        seen_keys.add(key)

    if len(rows_by_game) != expected_game_count:
        raise RollingFeatureError(
            f"Source contains {len(rows_by_game)} games, expected {expected_game_count}"
        )

    performances: list[dict[str, Any]] = []
    performance_lookup: dict[
        tuple[str, str],
        dict[str, Any],
    ] = {}

    for game_id, team_rows in rows_by_game.items():
        if len(team_rows) != 2:
            raise RollingFeatureError(f"Game {game_id} does not have two rows")

        for team_id, row in team_rows.items():
            opponent_team_id = str(row.get("opponent_team_id", "")).strip()
            opponent = team_rows.get(opponent_team_id)

            if opponent is None:
                raise RollingFeatureError(
                    f"Opponent row missing for game {game_id}, team {team_id}"
                )

            if str(opponent.get("opponent_team_id")) != team_id:
                raise RollingFeatureError(f"Opponent linkage is asymmetric for game {game_id}")

            scheduled_start = _parse_utc(row.get("scheduled_start_utc"))

            official_score_for = _integer(
                row,
                "official_final_score",
            )
            official_score_against = _integer(
                opponent,
                "official_final_score",
            )

            performance = {
                "game_id": game_id,
                "team_id": team_id,
                "opponent_team_id": opponent_team_id,
                "season_id": str(row["season_id"]),
                "scheduled_start_utc": (scheduled_start),
                "venue_side": str(row["venue_side"]),
                "win": int(official_score_for > official_score_against),
                "goals_for": _integer(
                    row,
                    "pbp_goals_non_shootout",
                ),
                "goals_against": _integer(
                    opponent,
                    "pbp_goals_non_shootout",
                ),
                "shots_on_goal_for": _integer(
                    row,
                    "official_shots_on_goal",
                ),
                "shots_on_goal_against": _integer(
                    opponent,
                    "official_shots_on_goal",
                ),
                "shot_attempts_for": _integer(
                    row,
                    "shot_attempt_events",
                ),
                "shot_attempts_against": _integer(
                    opponent,
                    "shot_attempt_events",
                ),
                "blocked_attempts_for": _integer(
                    row,
                    "blocked_shot_attempts",
                ),
                "blocked_attempts_against": _integer(
                    opponent,
                    "blocked_shot_attempts",
                ),
                "penalties_for": _integer(
                    row,
                    "penalties",
                ),
                "penalties_against": _integer(
                    opponent,
                    "penalties",
                ),
                "penalty_minutes_for": _integer(
                    row,
                    "penalty_minutes",
                ),
                "penalty_minutes_against": _integer(
                    opponent,
                    "penalty_minutes",
                ),
                "faceoff_wins_for": _integer(
                    row,
                    "faceoff_wins",
                ),
                "faceoff_wins_against": _integer(
                    opponent,
                    "faceoff_wins",
                ),
                "hits_for": _integer(
                    row,
                    "hits",
                ),
                "hits_against": _integer(
                    opponent,
                    "hits",
                ),
                "giveaways_for": _integer(
                    row,
                    "giveaways",
                ),
                "giveaways_against": _integer(
                    opponent,
                    "giveaways",
                ),
                "takeaways_for": _integer(
                    row,
                    "takeaways",
                ),
                "takeaways_against": _integer(
                    opponent,
                    "takeaways",
                ),
            }

            performances.append(performance)
            performance_lookup[(game_id, team_id)] = performance

    performances.sort(
        key=lambda row: (
            row["scheduled_start_utc"],
            row["game_id"],
            row["team_id"],
        )
    )

    history_groups: dict[
        tuple[str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for performance in performances:
        history_groups[
            (
                performance["team_id"],
                performance["season_id"],
            )
        ].append(performance)

    output_rows: list[dict[str, Any]] = []
    same_date_candidate_count = 0

    for current in performances:
        group = history_groups[
            (
                current["team_id"],
                current["season_id"],
            )
        ]

        same_date_candidates = [
            previous
            for previous in group
            if (
                previous["scheduled_start_utc"] < current["scheduled_start_utc"]
                and previous["scheduled_start_utc"].date() == current["scheduled_start_utc"].date()
            )
        ]

        same_date_candidate_count += len(same_date_candidates)

        eligible_history = [
            previous
            for previous in group
            if (previous["scheduled_start_utc"].date() < current["scheduled_start_utc"].date())
        ]

        eligible_history.sort(
            key=lambda row: (
                row["scheduled_start_utc"],
                row["game_id"],
            )
        )

        last_three = eligible_history[-3:]
        last_five = eligible_history[-5:]

        previous_game = eligible_history[-1] if eligible_history else None

        history_game_ids = [previous["game_id"] for previous in eligible_history]

        row = {
            "schema_version": "1.0",
            "source_batch_id": source_batch_id,
            "source_team_game_sha256": (source_sha256),
            "history_rule": HISTORY_RULE,
            "shots_on_goal_source": SHOTS_ON_GOAL_SOURCE,
            "goals_source": GOALS_SOURCE,
            "game_id": current["game_id"],
            "season_id": current["season_id"],
            "scheduled_start_utc": (current["scheduled_start_utc"].isoformat()),
            "team_id": current["team_id"],
            "opponent_team_id": (current["opponent_team_id"]),
            "venue_side": current["venue_side"],
            "history_game_ids_season_to_date": (history_game_ids),
            "history_game_ids_last_3": [previous["game_id"] for previous in last_three],
            "history_game_ids_last_5": [previous["game_id"] for previous in last_five],
            "previous_game_id": (previous_game["game_id"] if previous_game is not None else None),
            "previous_game_scheduled_start_utc": (
                previous_game["scheduled_start_utc"].isoformat()
                if previous_game is not None
                else None
            ),
            "days_since_previous_game_start": (
                (
                    current["scheduled_start_utc"] - previous_game["scheduled_start_utc"]
                ).total_seconds()
                / 86400.0
                if previous_game is not None
                else None
            ),
            "same_utc_date_prior_candidates_excluded": (len(same_date_candidates)),
            **_window_features(
                eligible_history,
                prefix="season_to_date",
            ),
            **_window_features(
                last_three,
                prefix="last_3",
            ),
            **_window_features(
                last_five,
                prefix="last_5",
            ),
        }

        output_rows.append(row)

    output_rows.sort(
        key=lambda row: (
            row["scheduled_start_utc"],
            row["game_id"],
            row["team_id"],
        )
    )

    current_game_reference_count = 0
    same_or_future_date_reference_count = 0
    cross_season_reference_count = 0
    wrong_team_reference_count = 0
    missing_history_reference_count = 0

    for row in output_rows:
        current_start = _parse_utc(row["scheduled_start_utc"])

        for history_game_id in row["history_game_ids_season_to_date"]:
            if history_game_id == row["game_id"]:
                current_game_reference_count += 1

            history = performance_lookup.get(
                (
                    history_game_id,
                    row["team_id"],
                )
            )

            if history is None:
                missing_history_reference_count += 1
                continue

            if history["team_id"] != row["team_id"]:
                wrong_team_reference_count += 1

            if history["season_id"] != row["season_id"]:
                cross_season_reference_count += 1

            if history["scheduled_start_utc"].date() >= current_start.date():
                same_or_future_date_reference_count += 1

    key_counts = Counter((row["game_id"], row["team_id"]) for row in output_rows)
    game_counts = Counter(row["game_id"] for row in output_rows)

    rows_with_any_history = sum(row["season_to_date_games"] >= 1 for row in output_rows)
    rows_with_three_history_games = sum(row["season_to_date_games"] >= 3 for row in output_rows)
    rows_with_five_history_games = sum(row["season_to_date_games"] >= 5 for row in output_rows)

    gates = {
        "passes_expected_game_count": (len(game_counts) == expected_game_count),
        "passes_expected_row_count": (len(output_rows) == expected_row_count),
        "passes_two_rows_per_game": all(count == 2 for count in game_counts.values()),
        "passes_unique_game_team_keys": all(count == 1 for count in key_counts.values()),
        "passes_no_current_game_reference": (current_game_reference_count == 0),
        "passes_prior_utc_date_only": (same_or_future_date_reference_count == 0),
        "passes_same_season_only": (cross_season_reference_count == 0),
        "passes_same_team_only": (wrong_team_reference_count == 0),
        "passes_complete_history_lineage": (missing_history_reference_count == 0),
    }

    status = "complete" if all(gates.values()) else "failed"

    output_data = _serialize_jsonl(output_rows)
    output_sha256 = _sha256(output_data)
    already_present = output_path.is_file()

    if already_present:
        if output_path.read_bytes() != output_data:
            raise RollingFeatureError(f"Existing rolling feature output differs: {output_path}")
    else:
        _write_bytes_atomically(
            output_path,
            output_data,
        )

    history_count_distribution = Counter(row["season_to_date_games"] for row in output_rows)

    audit = {
        "schema_version": "1.0",
        "source_batch_id": source_batch_id,
        "source_team_game_path": str(team_game_path),
        "source_team_game_sha256": (source_sha256),
        "history_rule": HISTORY_RULE,
        "shots_on_goal_source": SHOTS_ON_GOAL_SOURCE,
        "goals_source": GOALS_SOURCE,
        "status": status,
        "game_count": len(game_counts),
        "row_count": len(output_rows),
        "rows_with_any_history": (rows_with_any_history),
        "rows_with_three_history_games": (rows_with_three_history_games),
        "rows_with_five_history_games": (rows_with_five_history_games),
        "same_date_candidate_count": (same_date_candidate_count),
        "history_count_distribution": {
            str(key): value for key, value in sorted(history_count_distribution.items())
        },
        "lineage_violation_counts": {
            "current_game_reference_count": (current_game_reference_count),
            "same_or_future_date_reference_count": (same_or_future_date_reference_count),
            "cross_season_reference_count": (cross_season_reference_count),
            "wrong_team_reference_count": (wrong_team_reference_count),
            "missing_history_reference_count": (missing_history_reference_count),
        },
        "gates": gates,
        "output_path": str(output_path),
        "output_sha256": output_sha256,
    }

    _write_json_atomically(audit_path, audit)

    if status != "complete":
        failed_gates = sorted(gate for gate, passed in gates.items() if not passed)
        raise RollingFeatureError(f"Rolling feature gates failed: {failed_gates}")

    return RollingFeatureResult(
        source_batch_id=source_batch_id,
        game_count=len(game_counts),
        row_count=len(output_rows),
        rows_with_any_history=(rows_with_any_history),
        rows_with_three_history_games=(rows_with_three_history_games),
        rows_with_five_history_games=(rows_with_five_history_games),
        output_path=str(output_path),
        output_sha256=output_sha256,
        audit_path=str(audit_path),
        already_present=already_present,
        status=status,
    )


def result_as_dict(
    result: RollingFeatureResult,
) -> dict[str, Any]:
    """Convert a rolling feature result into a mapping."""
    return asdict(result)
