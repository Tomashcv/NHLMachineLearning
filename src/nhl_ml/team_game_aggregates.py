"""Build deterministic team-game aggregates from canonical NHL PBP events."""

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


class TeamGameAggregateError(ValueError):
    """Raised when team-game aggregates cannot be built safely."""


@dataclass(frozen=True, slots=True)
class TeamGameAggregateResult:
    """Summary of one team-game aggregate build."""

    batch_id: str
    source_selection_id: str
    game_count: int
    row_count: int
    unique_team_count: int
    canonical_event_count: int
    all_sog_reconciled: bool
    all_applicable_scores_reconciled: bool
    output_path: str
    output_sha256: str
    audit_path: str
    already_present: bool
    status: str


SHOT_ATTEMPT_TYPES = {
    "blocked-shot",
    "goal",
    "missed-shot",
    "shot-on-goal",
}

TRACKED_TEAM_EVENT_TYPES = {
    "blocked-shot",
    "delayed-penalty",
    "faceoff",
    "giveaway",
    "goal",
    "hit",
    "missed-shot",
    "penalty",
    "shot-on-goal",
    "takeaway",
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TeamGameAggregateError(f"Invalid JSON file: {path}") from exc

    if not isinstance(payload, dict):
        raise TeamGameAggregateError(f"Expected JSON object in {path}")

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
                raise TeamGameAggregateError(
                    f"Invalid JSONL in {path} at line {line_number}"
                ) from exc

            if not isinstance(record, dict):
                raise TeamGameAggregateError(
                    f"Expected JSON object in {path} at line {line_number}"
                )

            records.append(record)

    return records


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise TeamGameAggregateError(f"Invalid YAML file: {path}") from exc

    if not isinstance(payload, dict):
        raise TeamGameAggregateError(f"Expected YAML mapping in {path}")

    return payload


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


def _expected_outcome(game: dict[str, Any]) -> str:
    if game.get("went_to_shootout") is True:
        return "shootout"

    if game.get("went_to_overtime") is True:
        return "overtime_only"

    return "regulation"


def _new_team_metrics() -> dict[str, int]:
    return {
        "pbp_goals_non_shootout": 0,
        "shot_on_goal_events": 0,
        "goals_with_shot_type": 0,
        "goals_without_shot_type": 0,
        "pbp_shots_on_goal": 0,
        "shot_attempt_events": 0,
        "shot_attempts_with_coordinates": 0,
        "missed_shots": 0,
        "blocked_shot_attempts": 0,
        "penalties": 0,
        "penalty_minutes": 0,
        "penalties_without_duration": 0,
        "delayed_penalties": 0,
        "faceoff_wins": 0,
        "hits": 0,
        "giveaways": 0,
        "takeaways": 0,
        "empty_net_goal_candidates": 0,
        "regulation_owned_event_count": 0,
        "overtime_owned_event_count": 0,
        "shootout_owned_event_count": 0,
    }


def _aggregate_event(
    *,
    metrics: dict[str, int],
    event: dict[str, Any],
) -> None:
    event_type = str(event["event_type"])
    period_type = str(event["period_type"]).upper()

    if period_type == "REG":
        metrics["regulation_owned_event_count"] += 1
    elif period_type == "OT":
        metrics["overtime_owned_event_count"] += 1
    elif period_type == "SO":
        metrics["shootout_owned_event_count"] += 1

    if period_type == "SO":
        return

    if event_type == "goal":
        metrics["pbp_goals_non_shootout"] += 1

        if event.get("shot_type") is None:
            metrics["goals_without_shot_type"] += 1
        else:
            metrics["goals_with_shot_type"] += 1
            metrics["pbp_shots_on_goal"] += 1

        if event.get("empty_net_candidate") is True:
            metrics["empty_net_goal_candidates"] += 1

    elif event_type == "shot-on-goal":
        metrics["shot_on_goal_events"] += 1
        metrics["pbp_shots_on_goal"] += 1

    if event_type in SHOT_ATTEMPT_TYPES:
        metrics["shot_attempt_events"] += 1

        if event.get("x_coord") is not None and event.get("y_coord") is not None:
            metrics["shot_attempts_with_coordinates"] += 1

    if event_type == "missed-shot":
        metrics["missed_shots"] += 1
    elif event_type == "blocked-shot":
        metrics["blocked_shot_attempts"] += 1
    elif event_type == "penalty":
        metrics["penalties"] += 1

        duration = event.get("penalty_duration_minutes")

        if duration is None:
            metrics["penalties_without_duration"] += 1
        else:
            metrics["penalty_minutes"] += int(duration)

    elif event_type == "delayed-penalty":
        metrics["delayed_penalties"] += 1
    elif event_type == "faceoff":
        metrics["faceoff_wins"] += 1
    elif event_type == "hit":
        metrics["hits"] += 1
    elif event_type == "giveaway":
        metrics["giveaways"] += 1
    elif event_type == "takeaway":
        metrics["takeaways"] += 1


def build_team_game_aggregates(
    *,
    pilot_path: Path,
    pbp_manifest_path: Path,
    canonical_output_root: Path,
    per_game_audit_root: Path,
    output_path: Path,
    audit_path: Path,
) -> TeamGameAggregateResult:
    """Build two deterministic aggregate rows for every pilot game."""
    if not pilot_path.is_file():
        raise FileNotFoundError(f"Pilot file does not exist: {pilot_path}")

    if not pbp_manifest_path.is_file():
        raise FileNotFoundError(f"PBP manifest does not exist: {pbp_manifest_path}")

    manifest = _read_yaml(pbp_manifest_path)
    batch = manifest.get("batch")
    manifest_games = manifest.get("games")

    if not isinstance(batch, dict):
        raise TeamGameAggregateError("PBP manifest has no batch mapping")

    if not isinstance(manifest_games, list):
        raise TeamGameAggregateError("PBP manifest games must be a list")

    batch_id = str(batch.get("batch_id", "")).strip()
    source_selection_id = str(batch.get("source_selection_id", "")).strip()
    source_selection_sha256 = str(batch.get("source_selection_sha256", "")).strip()
    expected_game_count = int(batch.get("expected_game_count", 0))

    if not batch_id or not source_selection_id or not source_selection_sha256:
        raise TeamGameAggregateError("PBP manifest batch metadata is incomplete")

    pilot_data = pilot_path.read_bytes()

    if _sha256(pilot_data) != source_selection_sha256:
        raise TeamGameAggregateError("Pilot hash does not match PBP manifest")

    pilot_records = _read_jsonl(pilot_path)

    if len(pilot_records) != expected_game_count:
        raise TeamGameAggregateError(
            f"Pilot contains {len(pilot_records)} games, expected {expected_game_count}"
        )

    manifest_by_game_id: dict[str, dict[str, Any]] = {}

    for item in manifest_games:
        if not isinstance(item, dict):
            raise TeamGameAggregateError("PBP manifest game must be a mapping")

        game_id = str(item.get("game_id", "")).strip()

        if not game_id:
            raise TeamGameAggregateError("PBP manifest game has no game_id")

        if game_id in manifest_by_game_id:
            raise TeamGameAggregateError(f"Duplicate manifest game ID: {game_id}")

        manifest_by_game_id[game_id] = item

    if len(manifest_by_game_id) != expected_game_count:
        raise TeamGameAggregateError("Manifest game count does not match expected count")

    aggregate_rows: list[dict[str, Any]] = []
    game_audits: list[dict[str, Any]] = []
    seen_game_ids: set[str] = set()
    unique_team_ids: set[str] = set()

    total_canonical_events = 0
    unknown_owner_team_event_count = 0

    for record in pilot_records:
        game = record.get("game")

        if not isinstance(game, dict):
            raise TeamGameAggregateError("Pilot record has no game mapping")

        game_id = str(game.get("game_id", "")).strip()

        if not game_id:
            raise TeamGameAggregateError("Pilot game has no game_id")

        if game_id in seen_game_ids:
            raise TeamGameAggregateError(f"Duplicate pilot game ID: {game_id}")

        manifest_game = manifest_by_game_id.get(game_id)

        if manifest_game is None:
            raise TeamGameAggregateError(f"Pilot game missing from manifest: {game_id}")

        expected_outcome = _expected_outcome(game)

        if manifest_game.get("expected_outcome") != expected_outcome:
            raise TeamGameAggregateError(f"Outcome mismatch for game {game_id}")

        home_team_id = str(game.get("home_team_id", "")).strip()
        away_team_id = str(game.get("away_team_id", "")).strip()

        if not home_team_id or not away_team_id or home_team_id == away_team_id:
            raise TeamGameAggregateError(f"Invalid team IDs for game {game_id}")

        team_ids = {away_team_id, home_team_id}
        unique_team_ids.update(team_ids)

        canonical_audit_path = per_game_audit_root / f"pbp_canonical_{game_id}.json"
        canonical_audit = _read_json(canonical_audit_path)

        if str(canonical_audit.get("game_id")) != game_id:
            raise TeamGameAggregateError(f"Canonical audit game mismatch: {game_id}")

        output_relative_path = str(
            canonical_audit.get(
                "output_relative_path",
                "",
            )
        ).strip()
        expected_output_sha256 = str(canonical_audit.get("output_sha256", "")).strip()

        if not output_relative_path or not expected_output_sha256:
            raise TeamGameAggregateError(f"Canonical audit incomplete: {game_id}")

        canonical_path = canonical_output_root / output_relative_path

        if not canonical_path.is_file():
            raise FileNotFoundError(f"Canonical events do not exist: {canonical_path}")

        canonical_data = canonical_path.read_bytes()

        if _sha256(canonical_data) != expected_output_sha256:
            raise TeamGameAggregateError(f"Canonical hash mismatch: {game_id}")

        canonical_records = _read_jsonl(canonical_path)

        if len(canonical_records) != int(canonical_audit.get("event_count", -1)):
            raise TeamGameAggregateError(f"Canonical event count mismatch: {game_id}")

        total_canonical_events += len(canonical_records)

        metrics_by_team = {
            away_team_id: _new_team_metrics(),
            home_team_id: _new_team_metrics(),
        }

        game_unknown_owner_count = 0

        for canonical_record in canonical_records:
            event = canonical_record.get("event")

            if not isinstance(event, dict):
                raise TeamGameAggregateError(f"Canonical record has no event: {game_id}")

            if str(event.get("game_id")) != game_id:
                raise TeamGameAggregateError(f"Canonical event game mismatch: {game_id}")

            event_type = str(event.get("event_type", ""))
            owner_team_id_raw = event.get("event_owner_team_id")
            owner_team_id = (
                str(owner_team_id_raw).strip() if owner_team_id_raw is not None else None
            )

            if event_type not in TRACKED_TEAM_EVENT_TYPES:
                continue

            if owner_team_id not in team_ids:
                game_unknown_owner_count += 1
                continue

            _aggregate_event(
                metrics=metrics_by_team[owner_team_id],
                event=event,
            )

        unknown_owner_team_event_count += game_unknown_owner_count

        official_scores = {
            away_team_id: int(game["away_score"]),
            home_team_id: int(game["home_score"]),
        }

        official_sog_raw = canonical_audit.get("official_shots_on_goal")

        if not isinstance(official_sog_raw, dict):
            raise TeamGameAggregateError(f"Official SOG missing for game {game_id}")

        official_sog = {team_id: int(official_sog_raw[team_id]) for team_id in team_ids}

        score_reconciliation_applicable = expected_outcome != "shootout"

        game_score_reconciled = not score_reconciliation_applicable or all(
            metrics_by_team[team_id]["pbp_goals_non_shootout"] == official_scores[team_id]
            for team_id in team_ids
        )

        game_sog_reconciled = all(
            metrics_by_team[team_id]["pbp_shots_on_goal"] == official_sog[team_id]
            for team_id in team_ids
        )

        sides = (
            (
                "away",
                away_team_id,
                home_team_id,
            ),
            (
                "home",
                home_team_id,
                away_team_id,
            ),
        )

        for (
            venue_side,
            team_id,
            opponent_team_id,
        ) in sides:
            metrics = metrics_by_team[team_id]
            attempts = metrics["shot_attempt_events"]
            attempts_with_coordinates = metrics["shot_attempts_with_coordinates"]

            aggregate_rows.append(
                {
                    "schema_version": "1.1",
                    "batch_id": batch_id,
                    "source_selection_id": (source_selection_id),
                    "source_selection_sha256": (source_selection_sha256),
                    "game_id": game_id,
                    "season_id": str(game["season_id"]),
                    "game_type": str(game["game_type"]),
                    "scheduled_start_utc": str(game["scheduled_start_utc"]),
                    "expected_outcome": expected_outcome,
                    "team_id": team_id,
                    "opponent_team_id": opponent_team_id,
                    "venue_side": venue_side,
                    "official_final_score": (official_scores[team_id]),
                    "official_shots_on_goal": (official_sog[team_id]),
                    "score_reconciliation_applicable": (score_reconciliation_applicable),
                    "score_reconciliation_passed": (game_score_reconciled),
                    "sog_reconciliation_passed": (game_sog_reconciled),
                    **metrics,
                    "unblocked_shot_attempt_events": (
                        metrics["pbp_shots_on_goal"]
                        + metrics["goals_without_shot_type"]
                        + metrics["missed_shots"]
                    ),
                    "shot_coordinate_coverage": (
                        attempts_with_coordinates / attempts if attempts else None
                    ),
                    "source_canonical_relative_path": (output_relative_path),
                    "source_canonical_sha256": (expected_output_sha256),
                }
            )

        game_audits.append(
            {
                "game_id": game_id,
                "expected_outcome": expected_outcome,
                "canonical_event_count": len(canonical_records),
                "team_row_count": 2,
                "official_scores": official_scores,
                "pbp_goals_non_shootout": {
                    team_id: metrics_by_team[team_id]["pbp_goals_non_shootout"]
                    for team_id in sorted(team_ids)
                },
                "score_reconciliation_applicable": (score_reconciliation_applicable),
                "score_reconciliation_passed": (game_score_reconciled),
                "official_shots_on_goal": official_sog,
                "pbp_shots_on_goal": {
                    team_id: metrics_by_team[team_id]["pbp_shots_on_goal"]
                    for team_id in sorted(team_ids)
                },
                "sog_reconciliation_passed": (game_sog_reconciled),
                "unknown_owner_team_event_count": (game_unknown_owner_count),
            }
        )

        seen_game_ids.add(game_id)

    aggregate_rows.sort(
        key=lambda row: (
            row["scheduled_start_utc"],
            row["game_id"],
            0 if row["venue_side"] == "away" else 1,
        )
    )

    output_data = _serialize_jsonl(aggregate_rows)
    output_digest = _sha256(output_data)

    already_present = output_path.is_file()

    if already_present:
        if output_path.read_bytes() != output_data:
            raise TeamGameAggregateError(f"Existing aggregate output differs: {output_path}")
    else:
        _write_bytes_atomically(output_path, output_data)

    key_counts = Counter((row["game_id"], row["team_id"]) for row in aggregate_rows)
    game_row_counts = Counter(row["game_id"] for row in aggregate_rows)

    all_sog_reconciled = all(game["sog_reconciliation_passed"] for game in game_audits)
    all_applicable_scores_reconciled = all(
        game["score_reconciliation_passed"] for game in game_audits
    )

    shot_attempt_identity_passed = all(
        row["shot_attempt_events"]
        == (
            row["pbp_shots_on_goal"]
            + row["goals_without_shot_type"]
            + row["missed_shots"]
            + row["blocked_shot_attempts"]
        )
        for row in aggregate_rows
    )

    gates = {
        "passes_expected_game_count": (len(seen_game_ids) == expected_game_count),
        "passes_expected_row_count": (len(aggregate_rows) == expected_game_count * 2),
        "passes_two_rows_per_game": all(count == 2 for count in game_row_counts.values()),
        "passes_unique_game_team_keys": all(count == 1 for count in key_counts.values()),
        "passes_shot_attempt_identity": (shot_attempt_identity_passed),
        "passes_sog_reconciliation": (all_sog_reconciled),
        "passes_applicable_score_reconciliation": (all_applicable_scores_reconciled),
        "passes_known_event_owner_teams": (unknown_owner_team_event_count == 0),
    }

    status = "complete" if all(gates.values()) else "failed"

    totals = Counter()

    numeric_metric_fields = (
        "pbp_goals_non_shootout",
        "pbp_shots_on_goal",
        "shot_attempt_events",
        "missed_shots",
        "blocked_shot_attempts",
        "penalties",
        "penalty_minutes",
        "delayed_penalties",
        "faceoff_wins",
        "hits",
        "giveaways",
        "takeaways",
        "empty_net_goal_candidates",
        "goals_without_shot_type",
    )

    for row in aggregate_rows:
        for field in numeric_metric_fields:
            totals[field] += int(row[field])

    audit = {
        "schema_version": "1.1",
        "batch_id": batch_id,
        "source_selection_id": source_selection_id,
        "source_selection_sha256": (source_selection_sha256),
        "status": status,
        "game_count": len(seen_game_ids),
        "row_count": len(aggregate_rows),
        "unique_team_count": len(unique_team_ids),
        "canonical_event_count": total_canonical_events,
        "output_path": str(output_path),
        "output_sha256": output_digest,
        "unknown_owner_team_event_count": (unknown_owner_team_event_count),
        "gates": gates,
        "totals": dict(sorted(totals.items())),
        "games": game_audits,
    }

    _write_json_atomically(audit_path, audit)

    if status != "complete":
        failed_gates = sorted(gate for gate, passed in gates.items() if not passed)
        raise TeamGameAggregateError(f"Team-game aggregate gates failed: {failed_gates}")

    return TeamGameAggregateResult(
        batch_id=batch_id,
        source_selection_id=source_selection_id,
        game_count=len(seen_game_ids),
        row_count=len(aggregate_rows),
        unique_team_count=len(unique_team_ids),
        canonical_event_count=total_canonical_events,
        all_sog_reconciled=all_sog_reconciled,
        all_applicable_scores_reconciled=(all_applicable_scores_reconciled),
        output_path=str(output_path),
        output_sha256=output_digest,
        audit_path=str(audit_path),
        already_present=already_present,
        status=status,
    )


def result_as_dict(
    result: TeamGameAggregateResult,
) -> dict[str, Any]:
    """Convert a team-game aggregate result into a mapping."""
    return asdict(result)
