"""Build season-scale team-game aggregates from verified NHL PBP."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from nhl_ml.pbp_canonical import classify_sog_reconciliation
from nhl_ml.season_scale_pbp_batches import (
    SeasonScalePbpBatchError,
    verify_season_pbp_batch_config,
)
from nhl_ml.team_game_aggregates import (
    TRACKED_TEAM_EVENT_TYPES,
    TeamGameAggregateError,
    TeamGameAggregateResult,
    aggregate_team_event,
    new_team_metrics,
)


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
                raise TeamGameAggregateError(
                    f"Invalid JSONL in {path} at line {line_number}"
                ) from exc

            if not isinstance(record, dict):
                raise TeamGameAggregateError(f"Expected object in {path} at line {line_number}")

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


def build_season_scale_team_game_aggregates(
    *,
    season_id: str,
    config_path: Path,
    inventory_path: Path,
    inventory_audit_path: Path,
    canonical_output_root: Path,
    per_game_audit_root: Path,
    output_path: Path,
    audit_path: Path,
) -> TeamGameAggregateResult:
    """Build two verified team-game rows for one NHL season."""
    verification = verify_season_pbp_batch_config(
        season_id=season_id,
        config_path=config_path,
        inventory_path=inventory_path,
        inventory_audit_path=inventory_audit_path,
    )

    if verification["status"] != "verified":
        raise SeasonScalePbpBatchError(f"Season config is not verified: {season_id}")

    config = _read_yaml(config_path)
    batch = config.get("batch")
    config_games = config.get("games")

    if not isinstance(batch, dict):
        raise TeamGameAggregateError("Season config has no batch mapping")

    if not isinstance(config_games, list):
        raise TeamGameAggregateError("Season config games must be a list")

    batch_id = str(batch["batch_id"])
    corpus_id = str(batch["corpus_id"])
    split_role = str(batch["split_role"])
    source_selection_id = str(batch["source_selection_id"])
    source_selection_sha256 = str(batch["source_selection_sha256"])
    source_inventory_sha256 = str(batch["source_inventory_sha256"])
    expected_game_count = int(batch["expected_game_count"])

    inventory_data = inventory_path.read_bytes()

    if _sha256(inventory_data) != source_inventory_sha256:
        raise TeamGameAggregateError("Inventory hash does not match season config")

    inventory_rows = [
        row for row in _read_jsonl(inventory_path) if str(row["season_id"]) == season_id
    ]

    if len(inventory_rows) != expected_game_count:
        raise TeamGameAggregateError(
            f"Season inventory contains {len(inventory_rows)} games, expected {expected_game_count}"
        )

    inventory_by_game_id = {str(row["game_id"]): row for row in inventory_rows}
    config_by_game_id = {
        str(game["game_id"]): game for game in config_games if isinstance(game, dict)
    }

    if len(inventory_by_game_id) != expected_game_count:
        raise TeamGameAggregateError("Season inventory has duplicate game IDs")

    if len(config_by_game_id) != expected_game_count:
        raise TeamGameAggregateError("Season config has duplicate game IDs")

    if set(inventory_by_game_id) != set(config_by_game_id):
        raise TeamGameAggregateError("Season config and inventory game IDs differ")

    aggregate_rows: list[dict[str, Any]] = []
    game_audits: list[dict[str, Any]] = []
    unique_team_ids: set[str] = set()

    total_canonical_events = 0
    unknown_owner_team_event_count = 0

    for game_id in sorted(
        inventory_by_game_id,
        key=lambda value: (
            str(inventory_by_game_id[value]["scheduled_start_utc"]),
            value,
        ),
    ):
        record = inventory_by_game_id[game_id]
        configured_game = config_by_game_id[game_id]

        if str(record["split_role"]) != split_role:
            raise TeamGameAggregateError(f"Split-role mismatch for game {game_id}")

        expected_outcome = str(configured_game["expected_outcome"])

        if expected_outcome != str(record["expected_outcome"]):
            raise TeamGameAggregateError(f"Outcome mismatch for game {game_id}")

        away_team_id = str(record["away_team_id"])
        home_team_id = str(record["home_team_id"])

        if not away_team_id or not home_team_id or away_team_id == home_team_id:
            raise TeamGameAggregateError(f"Invalid team IDs for game {game_id}")

        team_ids = {
            away_team_id,
            home_team_id,
        }
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
        expected_output_sha256 = str(
            canonical_audit.get(
                "output_sha256",
                "",
            )
        ).strip()

        if not output_relative_path or not expected_output_sha256:
            raise TeamGameAggregateError(f"Canonical audit incomplete: {game_id}")

        canonical_path = canonical_output_root / output_relative_path

        if not canonical_path.is_file():
            raise FileNotFoundError(f"Canonical events do not exist: {canonical_path}")

        canonical_data = canonical_path.read_bytes()

        if _sha256(canonical_data) != expected_output_sha256:
            raise TeamGameAggregateError(f"Canonical hash mismatch: {game_id}")

        canonical_records = _read_jsonl(canonical_path)

        if len(canonical_records) != int(
            canonical_audit.get(
                "event_count",
                -1,
            )
        ):
            raise TeamGameAggregateError(f"Canonical event count mismatch: {game_id}")

        total_canonical_events += len(canonical_records)

        metrics_by_team = {
            away_team_id: new_team_metrics(),
            home_team_id: new_team_metrics(),
        }
        game_unknown_owner_count = 0

        for canonical_record in canonical_records:
            event = canonical_record.get("event")

            if not isinstance(event, dict):
                raise TeamGameAggregateError(f"Canonical record has no event: {game_id}")

            if str(event.get("game_id")) != game_id:
                raise TeamGameAggregateError(f"Canonical event game mismatch: {game_id}")

            event_type = str(event.get("event_type", ""))

            if event_type not in TRACKED_TEAM_EVENT_TYPES:
                continue

            owner_raw = event.get("event_owner_team_id")
            owner_team_id = str(owner_raw).strip() if owner_raw is not None else None

            if owner_team_id not in team_ids:
                game_unknown_owner_count += 1
                continue

            aggregate_team_event(
                metrics=metrics_by_team[owner_team_id],
                event=event,
            )

        unknown_owner_team_event_count += game_unknown_owner_count

        official_scores = {
            away_team_id: int(record["away_score"]),
            home_team_id: int(record["home_score"]),
        }
        inventory_official_sog = {
            away_team_id: int(record["away_shots_on_goal"]),
            home_team_id: int(record["home_shots_on_goal"]),
        }

        canonical_official_sog_raw = canonical_audit.get("official_shots_on_goal")

        if not isinstance(
            canonical_official_sog_raw,
            dict,
        ):
            raise TeamGameAggregateError(f"Official SOG missing: {game_id}")

        canonical_official_sog = {
            team_id: int(canonical_official_sog_raw[team_id]) for team_id in team_ids
        }

        if canonical_official_sog != inventory_official_sog:
            raise TeamGameAggregateError(f"Inventory and canonical official SOG differ: {game_id}")

        pbp_sog = {team_id: metrics_by_team[team_id]["pbp_shots_on_goal"] for team_id in team_ids}

        (
            game_sog_reconciled,
            sog_status,
            sog_deltas,
            correction_team_id,
        ) = classify_sog_reconciliation(
            inventory_official_sog,
            pbp_sog,
        )

        canonical_status = str(
            canonical_audit.get(
                "sog_reconciliation_status",
                "",
            )
        )
        canonical_deltas_raw = canonical_audit.get("sog_deltas_by_team")

        if not isinstance(
            canonical_deltas_raw,
            dict,
        ):
            raise TeamGameAggregateError(f"Canonical SOG deltas missing: {game_id}")

        canonical_deltas = {
            str(team_id): int(delta) for team_id, delta in canonical_deltas_raw.items()
        }
        canonical_correction_team_id_raw = canonical_audit.get("sog_provider_correction_team_id")
        canonical_correction_team_id = (
            str(canonical_correction_team_id_raw)
            if canonical_correction_team_id_raw is not None
            else None
        )

        if (
            canonical_status != sog_status
            or canonical_deltas != sog_deltas
            or canonical_correction_team_id != correction_team_id
            or bool(canonical_audit.get("sog_reconciliation_passed")) != game_sog_reconciled
        ):
            raise TeamGameAggregateError(f"Canonical SOG policy mismatch: {game_id}")

        score_reconciliation_applicable = expected_outcome != "shootout"
        game_score_reconciled = not score_reconciliation_applicable or all(
            metrics_by_team[team_id]["pbp_goals_non_shootout"] == official_scores[team_id]
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
                    "schema_version": "1.2",
                    "batch_id": batch_id,
                    "corpus_id": corpus_id,
                    "source_selection_id": (source_selection_id),
                    "source_selection_sha256": (source_selection_sha256),
                    "source_inventory_sha256": (source_inventory_sha256),
                    "split_role": split_role,
                    "game_id": game_id,
                    "season_id": season_id,
                    "game_type": "regular_season",
                    "scheduled_start_utc": str(record["scheduled_start_utc"]),
                    "expected_outcome": (expected_outcome),
                    "team_id": team_id,
                    "opponent_team_id": (opponent_team_id),
                    "venue_side": venue_side,
                    "official_final_score": (official_scores[team_id]),
                    "official_shots_on_goal": (inventory_official_sog[team_id]),
                    "score_reconciliation_applicable": (score_reconciliation_applicable),
                    "score_reconciliation_passed": (game_score_reconciled),
                    "sog_reconciliation_passed": (game_sog_reconciled),
                    "sog_reconciliation_status": (sog_status),
                    "sog_delta_pbp_minus_official": (sog_deltas[team_id]),
                    "sog_provider_correction_applied": (correction_team_id == team_id),
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
                "expected_outcome": (expected_outcome),
                "canonical_event_count": len(canonical_records),
                "team_row_count": 2,
                "official_scores": official_scores,
                "pbp_goals_non_shootout": {
                    team_id: metrics_by_team[team_id]["pbp_goals_non_shootout"]
                    for team_id in sorted(
                        team_ids,
                        key=int,
                    )
                },
                "score_reconciliation_applicable": (score_reconciliation_applicable),
                "score_reconciliation_passed": (game_score_reconciled),
                "official_shots_on_goal": (inventory_official_sog),
                "pbp_shots_on_goal": pbp_sog,
                "sog_deltas_by_team": (sog_deltas),
                "sog_reconciliation_status": (sog_status),
                "sog_provider_correction_team_id": (correction_team_id),
                "sog_reconciliation_passed": (game_sog_reconciled),
                "unknown_owner_team_event_count": (game_unknown_owner_count),
            }
        )

    aggregate_rows.sort(
        key=lambda row: (
            row["scheduled_start_utc"],
            row["game_id"],
            0 if row["venue_side"] == "away" else 1,
        )
    )

    output_data = _serialize_jsonl(aggregate_rows)
    output_sha256 = _sha256(output_data)
    already_present = output_path.is_file()

    if already_present:
        if output_path.read_bytes() != output_data:
            raise TeamGameAggregateError(f"Existing aggregate output differs: {output_path}")
    else:
        _write_bytes_atomically(
            output_path,
            output_data,
        )

    key_counts = Counter(
        (
            row["game_id"],
            row["team_id"],
        )
        for row in aggregate_rows
    )
    game_row_counts = Counter(row["game_id"] for row in aggregate_rows)
    sog_status_counts = Counter(game["sog_reconciliation_status"] for game in game_audits)

    all_sog_reconciled = all(game["sog_reconciliation_passed"] for game in game_audits)
    all_scores_reconciled = all(game["score_reconciliation_passed"] for game in game_audits)
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
        "passes_expected_game_count": (len(game_audits) == expected_game_count),
        "passes_expected_row_count": (len(aggregate_rows) == expected_game_count * 2),
        "passes_two_rows_per_game": all(count == 2 for count in game_row_counts.values()),
        "passes_unique_game_team_keys": all(count == 1 for count in key_counts.values()),
        "passes_shot_attempt_identity": (shot_attempt_identity_passed),
        "passes_sog_reconciliation_policy": (all_sog_reconciled),
        "passes_applicable_score_reconciliation": (all_scores_reconciled),
        "passes_known_event_owner_teams": (unknown_owner_team_event_count == 0),
    }

    status = "complete" if all(gates.values()) else "failed"

    numeric_metric_fields = (
        "pbp_goals_non_shootout",
        "official_shots_on_goal",
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
    totals: Counter[str] = Counter()

    for row in aggregate_rows:
        for field in numeric_metric_fields:
            totals[field] += int(row[field])

    audit = {
        "schema_version": "1.2",
        "batch_id": batch_id,
        "corpus_id": corpus_id,
        "season_id": season_id,
        "split_role": split_role,
        "source_selection_id": (source_selection_id),
        "source_selection_sha256": (source_selection_sha256),
        "source_inventory_sha256": (source_inventory_sha256),
        "status": status,
        "game_count": len(game_audits),
        "row_count": len(aggregate_rows),
        "unique_team_count": len(unique_team_ids),
        "canonical_event_count": (total_canonical_events),
        "sog_reconciliation_status_counts": dict(sorted(sog_status_counts.items())),
        "sog_provider_correction_game_count": (
            sog_status_counts["provider_boxscore_minus_one_correction"]
        ),
        "output_path": str(output_path),
        "output_sha256": output_sha256,
        "unknown_owner_team_event_count": (unknown_owner_team_event_count),
        "gates": gates,
        "totals": dict(sorted(totals.items())),
        "games": game_audits,
    }

    _write_json_atomically(
        audit_path,
        audit,
    )

    if status != "complete":
        failed_gates = sorted(gate for gate, passed in gates.items() if not passed)
        raise TeamGameAggregateError(f"Season team-game aggregate gates failed: {failed_gates}")

    return TeamGameAggregateResult(
        batch_id=batch_id,
        source_selection_id=(source_selection_id),
        game_count=len(game_audits),
        row_count=len(aggregate_rows),
        unique_team_count=len(unique_team_ids),
        canonical_event_count=(total_canonical_events),
        all_sog_reconciled=(all_sog_reconciled),
        all_applicable_scores_reconciled=(all_scores_reconciled),
        output_path=str(output_path),
        output_sha256=output_sha256,
        audit_path=str(audit_path),
        already_present=already_present,
        status=status,
    )
