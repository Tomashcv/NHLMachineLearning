"""Canonicalization and reconciliation of NHL play-by-play events."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


class PbpCanonicalizationError(ValueError):
    """Raised when PBP events cannot be canonicalized safely."""


@dataclass(frozen=True, slots=True)
class PbpCanonicalizationResult:
    """Summary of one canonical PBP conversion."""

    game_id: str
    raw_relative_path: str
    raw_sha256: str
    output_relative_path: str
    output_sha256: str
    event_count: int
    goal_count: int
    shot_on_goal_event_count: int
    shot_attempt_count: int
    penalty_count: int
    empty_net_goal_candidate_count: int
    score_reconciliation_passed: bool | None
    sog_reconciliation_passed: bool
    already_present: bool
    audit_path: str


SHOT_ATTEMPT_TYPES = {
    "blocked-shot",
    "goal",
    "missed-shot",
    "shot-on-goal",
}

SOG_EVENT_TYPES = {
    "goal",
    "shot-on-goal",
}

TEAM_OWNED_EVENT_TYPES = {
    "blocked-shot",
    "goal",
    "missed-shot",
    "penalty",
    "shot-on-goal",
}

PLAYER_DETAIL_FIELDS = {
    "assist1_player_id": "assist1PlayerId",
    "assist2_player_id": "assist2PlayerId",
    "blocking_player_id": "blockingPlayerId",
    "committed_by_player_id": "committedByPlayerId",
    "drawn_by_player_id": "drawnByPlayerId",
    "goalie_in_net_id": "goalieInNetId",
    "hittee_player_id": "hitteePlayerId",
    "hitting_player_id": "hittingPlayerId",
    "losing_player_id": "losingPlayerId",
    "player_id": "playerId",
    "scoring_player_id": "scoringPlayerId",
    "served_by_player_id": "servedByPlayerId",
    "shooting_player_id": "shootingPlayerId",
    "winning_player_id": "winningPlayerId",
}

MAPPED_DETAIL_FIELDS = {
    "assist1PlayerId",
    "assist2PlayerId",
    "awaySOG",
    "awayScore",
    "blockingPlayerId",
    "committedByPlayerId",
    "descKey",
    "drawnByPlayerId",
    "duration",
    "eventOwnerTeamId",
    "goalieInNetId",
    "hitteePlayerId",
    "hittingPlayerId",
    "homeSOG",
    "homeScore",
    "losingPlayerId",
    "playerId",
    "reason",
    "scoringPlayerId",
    "secondaryReason",
    "servedByPlayerId",
    "shootingPlayerId",
    "shotType",
    "typeCode",
    "winningPlayerId",
    "xCoord",
    "yCoord",
    "zoneCode",
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _require_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PbpCanonicalizationError(f"{field_name} must be a JSON object")

    return value


def _require_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PbpCanonicalizationError(f"{field_name} must be a non-empty string")

    return value.strip()


def _require_integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PbpCanonicalizationError(f"{field_name} must be an integer")

    return value


def _optional_integer(value: Any, field_name: str) -> int | None:
    if value is None:
        return None

    return _require_integer(value, field_name)


def _optional_identifier(value: Any) -> str | None:
    if value is None:
        return None

    parsed = str(value).strip()
    return parsed or None


def _optional_number(
    value: Any,
    field_name: str,
) -> int | float | None:
    if value is None:
        return None

    if isinstance(value, bool) or not isinstance(value, int | float):
        raise PbpCanonicalizationError(f"{field_name} must be numeric")

    return value


def _clock_to_seconds(value: Any, field_name: str) -> int:
    text = _require_non_empty_string(value, field_name)
    components = text.split(":")

    if len(components) != 2:
        raise PbpCanonicalizationError(f"{field_name} must use MM:SS format")

    try:
        minutes = int(components[0])
        seconds = int(components[1])
    except ValueError as exc:
        raise PbpCanonicalizationError(f"{field_name} must use MM:SS format") from exc

    if minutes < 0 or not 0 <= seconds < 60:
        raise PbpCanonicalizationError(f"{field_name} contains an invalid clock value")

    return minutes * 60 + seconds


def _parse_utc_timestamp(value: Any, field_name: str) -> str:
    text = _require_non_empty_string(value, field_name)
    normalized = text.replace("Z", "+00:00")

    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise PbpCanonicalizationError(f"{field_name} is not a valid ISO timestamp") from exc

    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise PbpCanonicalizationError(f"{field_name} must be timezone-aware")

    if parsed.utcoffset() != timedelta(0):
        raise PbpCanonicalizationError(f"{field_name} must be represented in UTC")

    return parsed.isoformat()


def _load_json_mapping(data: bytes, path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PbpCanonicalizationError(f"Raw PBP file is not valid UTF-8 JSON: {path}") from exc

    return _require_mapping(payload, "raw payload")


def _find_import_record(
    *,
    manifest_path: Path,
    raw_relative_path: str,
    raw_sha256: str,
    game_id: str,
) -> dict[str, Any]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Import manifest does not exist: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()

            if not stripped:
                continue

            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise PbpCanonicalizationError(
                    f"Invalid manifest JSONL at line {line_number}"
                ) from exc

            if not isinstance(record, dict):
                continue

            if (
                record.get("endpoint_family") == "play_by_play"
                and record.get("raw_relative_path") == raw_relative_path
                and record.get("sha256") == raw_sha256
                and str(record.get("game_id")) == game_id
            ):
                return record

    raise PbpCanonicalizationError("No matching play-by-play import record was found")


def _team_summary(
    payload: dict[str, Any],
    field_name: str,
) -> dict[str, Any]:
    team = _require_mapping(payload.get(field_name), field_name)

    team_id = _optional_identifier(team.get("id"))

    if team_id is None:
        raise PbpCanonicalizationError(f"{field_name}.id is required")

    return {
        "team_id": team_id,
        "abbreviation": _optional_identifier(team.get("abbrev")),
        "score": _optional_integer(
            team.get("score"),
            f"{field_name}.score",
        ),
        "shots_on_goal": _optional_integer(
            team.get("sog"),
            f"{field_name}.sog",
        ),
    }


def _canonicalize_play(
    *,
    game_id: str,
    play: dict[str, Any],
    source_index: int,
    source_observed_at_utc: str,
) -> dict[str, Any]:
    event_id = _optional_identifier(play.get("eventId"))

    if event_id is None:
        raise PbpCanonicalizationError(f"plays[{source_index}].eventId is required")

    sort_order = _require_integer(
        play.get("sortOrder"),
        f"plays[{source_index}].sortOrder",
    )
    event_type = _require_non_empty_string(
        play.get("typeDescKey"),
        f"plays[{source_index}].typeDescKey",
    )

    period = _require_mapping(
        play.get("periodDescriptor"),
        f"plays[{source_index}].periodDescriptor",
    )
    period_number = _require_integer(
        period.get("number"),
        f"plays[{source_index}].periodDescriptor.number",
    )
    period_type = _require_non_empty_string(
        period.get("periodType"),
        f"plays[{source_index}].periodDescriptor.periodType",
    )

    time_in_period = _require_non_empty_string(
        play.get("timeInPeriod"),
        f"plays[{source_index}].timeInPeriod",
    )
    period_elapsed_seconds = _clock_to_seconds(
        time_in_period,
        f"plays[{source_index}].timeInPeriod",
    )

    time_remaining_raw = play.get("timeRemaining")
    time_remaining = None
    period_remaining_seconds = None

    if time_remaining_raw is not None:
        time_remaining = _require_non_empty_string(
            time_remaining_raw,
            f"plays[{source_index}].timeRemaining",
        )
        period_remaining_seconds = _clock_to_seconds(
            time_remaining,
            f"plays[{source_index}].timeRemaining",
        )

    details_raw = play.get("details")

    if details_raw is None:
        details: dict[str, Any] = {}
    else:
        details = _require_mapping(
            details_raw,
            f"plays[{source_index}].details",
        )

    player_ids = {
        canonical_name: _optional_identifier(details.get(provider_name))
        for canonical_name, provider_name in PLAYER_DETAIL_FIELDS.items()
    }

    goalie_in_net_id = player_ids["goalie_in_net_id"]

    return {
        "game_id": game_id,
        "event_id": event_id,
        "source_index": source_index,
        "sort_order": sort_order,
        "event_type": event_type,
        "period_number": period_number,
        "period_type": period_type,
        "time_in_period": time_in_period,
        "period_elapsed_seconds": period_elapsed_seconds,
        "time_remaining": time_remaining,
        "period_remaining_seconds": period_remaining_seconds,
        "situation_code": _optional_identifier(play.get("situationCode")),
        "home_team_defending_side": _optional_identifier(play.get("homeTeamDefendingSide")),
        "event_owner_team_id": _optional_identifier(details.get("eventOwnerTeamId")),
        "x_coord": _optional_number(
            details.get("xCoord"),
            f"plays[{source_index}].details.xCoord",
        ),
        "y_coord": _optional_number(
            details.get("yCoord"),
            f"plays[{source_index}].details.yCoord",
        ),
        "zone_code": _optional_identifier(details.get("zoneCode")),
        "shot_type": _optional_identifier(details.get("shotType")),
        "reason": _optional_identifier(details.get("reason")),
        "secondary_reason": _optional_identifier(details.get("secondaryReason")),
        "penalty_description": _optional_identifier(details.get("descKey")),
        "penalty_type_code": _optional_identifier(details.get("typeCode")),
        "penalty_duration_minutes": _optional_integer(
            details.get("duration"),
            f"plays[{source_index}].details.duration",
        ),
        "home_score_snapshot": _optional_integer(
            details.get("homeScore"),
            f"plays[{source_index}].details.homeScore",
        ),
        "away_score_snapshot": _optional_integer(
            details.get("awayScore"),
            f"plays[{source_index}].details.awayScore",
        ),
        "home_sog_snapshot": _optional_integer(
            details.get("homeSOG"),
            f"plays[{source_index}].details.homeSOG",
        ),
        "away_sog_snapshot": _optional_integer(
            details.get("awaySOG"),
            f"plays[{source_index}].details.awaySOG",
        ),
        "player_ids": player_ids,
        "empty_net_candidate": (event_type == "goal" and goalie_in_net_id is None),
        "unmapped_detail_keys": sorted(set(details) - MAPPED_DETAIL_FIELDS),
        "source": "nhl_web",
        "source_observed_at_utc": source_observed_at_utc,
    }


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


def canonicalize_pbp_file(
    *,
    raw_file: Path,
    raw_root: Path,
    import_manifest_path: Path,
    output_root: Path,
    audit_path: Path,
) -> PbpCanonicalizationResult:
    """Canonicalize one imported NHL play-by-play file."""
    raw_root = raw_root.expanduser().resolve()
    raw_file = raw_file.expanduser().resolve()

    if not raw_file.is_file():
        raise FileNotFoundError(f"Raw PBP file does not exist: {raw_file}")

    try:
        raw_relative_path = raw_file.relative_to(raw_root).as_posix()
    except ValueError as exc:
        raise PbpCanonicalizationError(f"Raw PBP file is outside raw root: {raw_file}") from exc

    raw_data = raw_file.read_bytes()
    raw_digest = _sha256(raw_data)
    payload = _load_json_mapping(raw_data, raw_file)

    game_id = _optional_identifier(payload.get("id"))

    if game_id is None:
        raise PbpCanonicalizationError("PBP payload is missing top-level id")

    import_record = _find_import_record(
        manifest_path=import_manifest_path,
        raw_relative_path=raw_relative_path,
        raw_sha256=raw_digest,
        game_id=game_id,
    )
    source_observed_at_utc = _parse_utc_timestamp(
        import_record.get("imported_at_utc"),
        "manifest.imported_at_utc",
    )

    home_team = _team_summary(payload, "homeTeam")
    away_team = _team_summary(payload, "awayTeam")

    plays_raw = payload.get("plays")

    if not isinstance(plays_raw, list):
        raise PbpCanonicalizationError("PBP payload must contain a plays list")

    canonical_events: list[dict[str, Any]] = []
    seen_event_ids: set[str] = set()
    previous_sort_order: int | None = None

    for source_index, play_raw in enumerate(plays_raw):
        play = _require_mapping(
            play_raw,
            f"plays[{source_index}]",
        )
        event = _canonicalize_play(
            game_id=game_id,
            play=play,
            source_index=source_index,
            source_observed_at_utc=source_observed_at_utc,
        )

        event_id = event["event_id"]

        if event_id in seen_event_ids:
            raise PbpCanonicalizationError(f"Duplicate event ID: {event_id}")

        seen_event_ids.add(event_id)

        sort_order = event["sort_order"]

        if previous_sort_order is not None and sort_order < previous_sort_order:
            raise PbpCanonicalizationError("PBP sortOrder is not nondecreasing")

        previous_sort_order = sort_order

        canonical_events.append(
            {
                "schema_version": "1.0",
                "source_id": "nhl_web",
                "source_raw_relative_path": raw_relative_path,
                "source_raw_sha256": raw_digest,
                "temporal_classification": "in_game_event_data",
                "event": event,
            }
        )

    output_data = _serialize_jsonl(canonical_events)
    output_digest = _sha256(output_data)

    output_relative_path = Path("nhl_web") / game_id / f"events_{raw_digest[:12]}.jsonl"
    destination = output_root / output_relative_path

    already_present = destination.is_file()

    if already_present:
        if destination.read_bytes() != output_data:
            raise PbpCanonicalizationError(
                f"Existing canonical output is not deterministic: {destination}"
            )
    else:
        _write_bytes_atomically(destination, output_data)

    event_type_counts: Counter[str] = Counter()
    goal_counts: Counter[str] = Counter()
    sog_counts: Counter[str] = Counter()
    unknown_detail_key_counts: Counter[str] = Counter()

    shot_attempt_count = 0
    shot_attempts_with_coordinates = 0
    penalty_count = 0
    penalty_minutes = 0
    empty_net_goal_candidate_count = 0
    missing_owner_team_core_event_count = 0
    contains_shootout_period = False

    for record in canonical_events:
        event = record["event"]
        event_type = event["event_type"]
        period_type = event["period_type"]
        team_id = event["event_owner_team_id"]

        event_type_counts[event_type] += 1
        unknown_detail_key_counts.update(event["unmapped_detail_keys"])

        if period_type == "SO":
            contains_shootout_period = True

        if event_type in SHOT_ATTEMPT_TYPES:
            shot_attempt_count += 1

            if event["x_coord"] is not None and event["y_coord"] is not None:
                shot_attempts_with_coordinates += 1

        if event_type in TEAM_OWNED_EVENT_TYPES and team_id is None:
            missing_owner_team_core_event_count += 1

        if period_type != "SO" and event_type == "goal" and team_id is not None:
            goal_counts[team_id] += 1

        if period_type != "SO" and event_type in SOG_EVENT_TYPES and team_id is not None:
            sog_counts[team_id] += 1

        if event_type == "penalty":
            penalty_count += 1
            penalty_minutes += event["penalty_duration_minutes"] or 0

        if event["empty_net_candidate"]:
            empty_net_goal_candidate_count += 1

    official_scores = {
        away_team["team_id"]: away_team["score"],
        home_team["team_id"]: home_team["score"],
    }
    official_sog = {
        away_team["team_id"]: away_team["shots_on_goal"],
        home_team["team_id"]: home_team["shots_on_goal"],
    }

    pbp_goal_counts = {team_id: goal_counts[team_id] for team_id in official_scores}
    pbp_sog_counts = {team_id: sog_counts[team_id] for team_id in official_sog}

    score_reconciliation_passed: bool | None

    if contains_shootout_period:
        score_reconciliation_passed = None
    else:
        score_reconciliation_passed = official_scores == pbp_goal_counts

    sog_reconciliation_passed = official_sog == pbp_sog_counts

    audit = {
        "schema_version": "1.0",
        "source_id": "nhl_web",
        "game_id": game_id,
        "raw_relative_path": raw_relative_path,
        "raw_sha256": raw_digest,
        "output_relative_path": output_relative_path.as_posix(),
        "output_sha256": output_digest,
        "event_count": len(canonical_events),
        "event_type_counts": dict(sorted(event_type_counts.items())),
        "shot_attempt_count": shot_attempt_count,
        "shot_attempts_with_coordinates": (shot_attempts_with_coordinates),
        "shot_coordinate_coverage": (
            shot_attempts_with_coordinates / shot_attempt_count if shot_attempt_count else None
        ),
        "goal_event_count": event_type_counts["goal"],
        "shot_on_goal_event_count": (event_type_counts["shot-on-goal"]),
        "penalty_count": penalty_count,
        "penalty_minutes": penalty_minutes,
        "empty_net_goal_candidate_count": (empty_net_goal_candidate_count),
        "missing_owner_team_core_event_count": (missing_owner_team_core_event_count),
        "contains_shootout_period": contains_shootout_period,
        "official_scores": official_scores,
        "pbp_goal_counts_excluding_shootout": pbp_goal_counts,
        "score_reconciliation_passed": (score_reconciliation_passed),
        "official_shots_on_goal": official_sog,
        "pbp_shots_on_goal_including_goals": pbp_sog_counts,
        "sog_reconciliation_passed": sog_reconciliation_passed,
        "unmapped_detail_key_counts": dict(sorted(unknown_detail_key_counts.items())),
    }

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

    return PbpCanonicalizationResult(
        game_id=game_id,
        raw_relative_path=raw_relative_path,
        raw_sha256=raw_digest,
        output_relative_path=output_relative_path.as_posix(),
        output_sha256=output_digest,
        event_count=len(canonical_events),
        goal_count=event_type_counts["goal"],
        shot_on_goal_event_count=event_type_counts["shot-on-goal"],
        shot_attempt_count=shot_attempt_count,
        penalty_count=penalty_count,
        empty_net_goal_candidate_count=(empty_net_goal_candidate_count),
        score_reconciliation_passed=(score_reconciliation_passed),
        sog_reconciliation_passed=sog_reconciliation_passed,
        already_present=already_present,
        audit_path=str(audit_path),
    )


def result_as_dict(
    result: PbpCanonicalizationResult,
) -> dict[str, Any]:
    """Convert a canonicalization result to a mapping."""
    return asdict(result)
