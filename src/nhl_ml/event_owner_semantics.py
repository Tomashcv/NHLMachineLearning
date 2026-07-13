"""Infer eventOwnerTeamId semantics from NHL player-team mappings."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


class EventOwnerSemanticError(ValueError):
    """Raised when event-owner semantics cannot be inferred safely."""


@dataclass(frozen=True, slots=True)
class EventOwnerSemanticResult:
    """Summary of one event-owner semantic audit."""

    batch_id: str
    game_count: int
    validation_row_count: int
    event_type_count: int
    all_player_ids_present: bool
    all_players_mapped: bool
    all_owner_teams_valid: bool
    all_semantics_unambiguous: bool
    output_path: str
    output_sha256: str
    audit_path: str
    already_present: bool
    status: str


EVENT_ROLE_FIELDS: dict[str, tuple[str, ...]] = {
    "blocked-shot": (
        "shooting_player_id",
        "blocking_player_id",
    ),
    "faceoff": (
        "winning_player_id",
        "losing_player_id",
    ),
    "hit": (
        "hitting_player_id",
        "hittee_player_id",
    ),
    "giveaway": ("player_id",),
    "takeaway": ("player_id",),
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EventOwnerSemanticError(f"Invalid JSON file: {path}") from exc

    if not isinstance(payload, dict):
        raise EventOwnerSemanticError(f"Expected JSON object in {path}")

    return payload


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise EventOwnerSemanticError(f"Invalid YAML file: {path}") from exc

    if not isinstance(payload, dict):
        raise EventOwnerSemanticError(f"Expected YAML mapping in {path}")

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
                raise EventOwnerSemanticError(
                    f"Invalid JSONL in {path} at line {line_number}"
                ) from exc

            if not isinstance(record, dict):
                raise EventOwnerSemanticError(
                    f"Expected JSON object in {path} at line {line_number}"
                )

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


def _identifier(value: Any) -> str | None:
    if value is None:
        return None

    parsed = str(value).strip()
    return parsed or None


def _build_roster_map(
    *,
    payload: dict[str, Any],
    game_id: str,
) -> tuple[dict[str, str], set[str]]:
    home_team = payload.get("homeTeam")
    away_team = payload.get("awayTeam")

    if not isinstance(home_team, dict):
        raise EventOwnerSemanticError(f"homeTeam missing for game {game_id}")

    if not isinstance(away_team, dict):
        raise EventOwnerSemanticError(f"awayTeam missing for game {game_id}")

    home_team_id = _identifier(home_team.get("id"))
    away_team_id = _identifier(away_team.get("id"))

    if home_team_id is None or away_team_id is None or home_team_id == away_team_id:
        raise EventOwnerSemanticError(f"Invalid home/away teams for game {game_id}")

    valid_team_ids = {
        home_team_id,
        away_team_id,
    }

    roster_spots = payload.get("rosterSpots")

    if not isinstance(roster_spots, list):
        raise EventOwnerSemanticError(f"rosterSpots missing for game {game_id}")

    roster_map: dict[str, str] = {}

    for index, roster_spot in enumerate(roster_spots):
        if not isinstance(roster_spot, dict):
            raise EventOwnerSemanticError(
                f"rosterSpots[{index}] is not an object for game {game_id}"
            )

        player_id = _identifier(roster_spot.get("playerId"))
        team_id = _identifier(roster_spot.get("teamId"))

        if player_id is None or team_id is None:
            raise EventOwnerSemanticError(
                f"Incomplete roster spot at index {index} for game {game_id}"
            )

        if team_id not in valid_team_ids:
            raise EventOwnerSemanticError(
                f"Roster player {player_id} has unknown team {team_id} in game {game_id}"
            )

        existing_team_id = roster_map.get(player_id)

        if existing_team_id is not None and existing_team_id != team_id:
            raise EventOwnerSemanticError(
                f"Conflicting roster teams for player {player_id} in game {game_id}"
            )

        roster_map[player_id] = team_id

    if not roster_map:
        raise EventOwnerSemanticError(f"Empty roster map for game {game_id}")

    return roster_map, valid_team_ids


def audit_event_owner_semantics(
    *,
    pbp_manifest_path: Path,
    raw_root: Path,
    canonical_output_root: Path,
    per_game_audit_root: Path,
    output_path: Path,
    audit_path: Path,
) -> EventOwnerSemanticResult:
    """Infer event-owner meaning from event player roles."""
    if not pbp_manifest_path.is_file():
        raise FileNotFoundError(f"PBP manifest does not exist: {pbp_manifest_path}")

    manifest = _read_yaml(pbp_manifest_path)
    batch = manifest.get("batch")
    manifest_games = manifest.get("games")

    if not isinstance(batch, dict):
        raise EventOwnerSemanticError("Manifest has no batch mapping")

    if not isinstance(manifest_games, list):
        raise EventOwnerSemanticError("Manifest games must be a list")

    batch_id = str(batch.get("batch_id", "")).strip()
    expected_game_count = int(batch.get("expected_game_count", 0))

    if not batch_id or expected_game_count <= 0:
        raise EventOwnerSemanticError("Manifest batch metadata is incomplete")

    validation_rows: list[dict[str, Any]] = []
    seen_game_ids: set[str] = set()

    event_type_counts: Counter[str] = Counter()
    owner_valid_counts: Counter[str] = Counter()
    all_role_ids_present_counts: Counter[str] = Counter()
    all_role_players_mapped_counts: Counter[str] = Counter()
    distinct_role_team_counts: Counter[str] = Counter()

    role_match_counts: dict[str, Counter[str]] = defaultdict(Counter)

    game_summaries: list[dict[str, Any]] = []

    for manifest_game in manifest_games:
        if not isinstance(manifest_game, dict):
            raise EventOwnerSemanticError("Manifest game is not a mapping")

        game_id = str(manifest_game.get("game_id", "")).strip()

        if not game_id:
            raise EventOwnerSemanticError("Manifest game has no game_id")

        if game_id in seen_game_ids:
            raise EventOwnerSemanticError(f"Duplicate game ID: {game_id}")

        canonical_audit_path = per_game_audit_root / f"pbp_canonical_{game_id}.json"
        canonical_audit = _read_json(canonical_audit_path)

        if str(canonical_audit.get("game_id")) != game_id:
            raise EventOwnerSemanticError(f"Canonical audit game mismatch: {game_id}")

        raw_relative_path = str(
            canonical_audit.get(
                "raw_relative_path",
                "",
            )
        ).strip()
        raw_sha256 = str(
            canonical_audit.get(
                "raw_sha256",
                "",
            )
        ).strip()
        canonical_relative_path = str(
            canonical_audit.get(
                "output_relative_path",
                "",
            )
        ).strip()
        canonical_sha256 = str(
            canonical_audit.get(
                "output_sha256",
                "",
            )
        ).strip()

        if not all(
            (
                raw_relative_path,
                raw_sha256,
                canonical_relative_path,
                canonical_sha256,
            )
        ):
            raise EventOwnerSemanticError(f"Incomplete lineage for game {game_id}")

        raw_path = raw_root / raw_relative_path
        canonical_path = canonical_output_root / canonical_relative_path

        if not raw_path.is_file():
            raise FileNotFoundError(f"Raw PBP file missing: {raw_path}")

        if not canonical_path.is_file():
            raise FileNotFoundError(f"Canonical PBP missing: {canonical_path}")

        raw_data = raw_path.read_bytes()
        canonical_data = canonical_path.read_bytes()

        if _sha256(raw_data) != raw_sha256:
            raise EventOwnerSemanticError(f"Raw hash mismatch for game {game_id}")

        if _sha256(canonical_data) != canonical_sha256:
            raise EventOwnerSemanticError(f"Canonical hash mismatch for game {game_id}")

        try:
            raw_payload = json.loads(raw_data)
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as exc:
            raise EventOwnerSemanticError(f"Invalid raw PBP JSON for game {game_id}") from exc

        if not isinstance(raw_payload, dict):
            raise EventOwnerSemanticError(f"Raw PBP is not an object for game {game_id}")

        roster_map, valid_team_ids = _build_roster_map(
            payload=raw_payload,
            game_id=game_id,
        )

        canonical_records = _read_jsonl(canonical_path)

        game_relevant_event_count = 0
        game_invalid_owner_count = 0
        game_missing_player_id_count = 0
        game_unmapped_player_count = 0
        game_non_distinct_role_team_count = 0

        for canonical_record in canonical_records:
            event = canonical_record.get("event")

            if not isinstance(event, dict):
                raise EventOwnerSemanticError(f"Canonical record has no event for game {game_id}")

            event_type = str(event.get("event_type", ""))

            role_fields = EVENT_ROLE_FIELDS.get(event_type)

            if role_fields is None:
                continue

            game_relevant_event_count += 1
            event_type_counts[event_type] += 1

            owner_team_id = _identifier(event.get("event_owner_team_id"))
            owner_team_valid = owner_team_id in valid_team_ids

            if owner_team_valid:
                owner_valid_counts[event_type] += 1
            else:
                game_invalid_owner_count += 1

            player_ids = event.get("player_ids")

            if not isinstance(player_ids, dict):
                player_ids = {}

            role_player_ids: dict[str, str | None] = {}
            role_team_ids: dict[str, str | None] = {}
            matching_roles: list[str] = []

            for role_field in role_fields:
                player_id = _identifier(player_ids.get(role_field))
                team_id = roster_map.get(player_id) if player_id is not None else None

                role_player_ids[role_field] = player_id
                role_team_ids[role_field] = team_id

                if owner_team_id is not None and team_id == owner_team_id:
                    matching_roles.append(role_field)
                    role_match_counts[event_type][role_field] += 1

            all_role_ids_present = all(
                role_player_ids[role_field] is not None for role_field in role_fields
            )

            if all_role_ids_present:
                all_role_ids_present_counts[event_type] += 1
            else:
                game_missing_player_id_count += 1

            all_role_players_mapped = all(
                role_team_ids[role_field] is not None for role_field in role_fields
            )

            if all_role_players_mapped:
                all_role_players_mapped_counts[event_type] += 1
            else:
                game_unmapped_player_count += 1

            distinct_role_teams: bool | None

            if len(role_fields) == 1:
                distinct_role_teams = None
            else:
                mapped_teams = {
                    role_team_ids[role_field]
                    for role_field in role_fields
                    if role_team_ids[role_field] is not None
                }
                distinct_role_teams = len(mapped_teams) == len(role_fields)

                if distinct_role_teams:
                    distinct_role_team_counts[event_type] += 1
                else:
                    game_non_distinct_role_team_count += 1

            validation_rows.append(
                {
                    "schema_version": "1.0",
                    "batch_id": batch_id,
                    "game_id": game_id,
                    "event_id": str(event.get("event_id", "")),
                    "event_type": event_type,
                    "period_number": event.get("period_number"),
                    "period_type": event.get("period_type"),
                    "time_in_period": event.get("time_in_period"),
                    "owner_team_id": owner_team_id,
                    "owner_team_valid": owner_team_valid,
                    "role_player_ids": (role_player_ids),
                    "role_team_ids": role_team_ids,
                    "all_role_ids_present": (all_role_ids_present),
                    "all_role_players_mapped": (all_role_players_mapped),
                    "distinct_role_teams": (distinct_role_teams),
                    "matching_roles": sorted(matching_roles),
                    "source_raw_relative_path": (raw_relative_path),
                    "source_raw_sha256": raw_sha256,
                    "source_canonical_relative_path": (canonical_relative_path),
                    "source_canonical_sha256": (canonical_sha256),
                }
            )

        game_summaries.append(
            {
                "game_id": game_id,
                "relevant_event_count": (game_relevant_event_count),
                "invalid_owner_count": (game_invalid_owner_count),
                "missing_player_id_count": (game_missing_player_id_count),
                "unmapped_player_count": (game_unmapped_player_count),
                "non_distinct_role_team_count": (game_non_distinct_role_team_count),
                "roster_player_count": len(roster_map),
            }
        )

        seen_game_ids.add(game_id)

    semantic_summaries: dict[str, dict[str, Any]] = {}

    for event_type, role_fields in sorted(EVENT_ROLE_FIELDS.items()):
        event_count = event_type_counts[event_type]

        perfect_matching_roles = [
            role_field
            for role_field in role_fields
            if (event_count > 0 and role_match_counts[event_type][role_field] == event_count)
        ]

        inferred_owner_role = (
            perfect_matching_roles[0] if len(perfect_matching_roles) == 1 else None
        )

        if event_count == 0:
            semantic_status = "no_events"
        elif inferred_owner_role is None:
            semantic_status = "ambiguous"
        else:
            semantic_status = "unambiguous"

        semantic_summaries[event_type] = {
            "event_count": event_count,
            "candidate_roles": list(role_fields),
            "role_match_counts": {
                role_field: role_match_counts[event_type][role_field] for role_field in role_fields
            },
            "owner_team_valid_count": (owner_valid_counts[event_type]),
            "all_role_ids_present_count": (all_role_ids_present_counts[event_type]),
            "all_role_players_mapped_count": (all_role_players_mapped_counts[event_type]),
            "distinct_role_team_count": (
                distinct_role_team_counts[event_type] if len(role_fields) > 1 else None
            ),
            "same_team_multi_role_event_count": (
                event_count - distinct_role_team_counts[event_type]
                if len(role_fields) > 1
                else None
            ),
            "inferred_owner_role": (inferred_owner_role),
            "semantic_status": semantic_status,
        }

    validation_rows.sort(
        key=lambda row: (
            row["game_id"],
            row["event_type"],
            row["period_number"],
            row["event_id"],
        )
    )

    output_data = _serialize_jsonl(validation_rows)
    output_digest = _sha256(output_data)

    already_present = output_path.is_file()

    if already_present:
        if output_path.read_bytes() != output_data:
            raise EventOwnerSemanticError(f"Existing semantic output differs: {output_path}")
    else:
        _write_bytes_atomically(
            output_path,
            output_data,
        )

    all_player_ids_present = all(
        event_type_counts[event_type] > 0
        and all_role_ids_present_counts[event_type] == event_type_counts[event_type]
        for event_type in EVENT_ROLE_FIELDS
    )

    all_players_mapped = all(
        event_type_counts[event_type] > 0
        and all_role_players_mapped_counts[event_type] == event_type_counts[event_type]
        for event_type in EVENT_ROLE_FIELDS
    )

    all_owner_teams_valid = all(
        event_type_counts[event_type] > 0
        and owner_valid_counts[event_type] == event_type_counts[event_type]
        for event_type in EVENT_ROLE_FIELDS
    )

    all_semantics_unambiguous = all(
        summary["semantic_status"] == "unambiguous" for summary in semantic_summaries.values()
    )

    gates = {
        "passes_expected_game_count": (len(seen_game_ids) == expected_game_count),
        "passes_required_event_type_coverage": all(
            event_type_counts[event_type] > 0 for event_type in EVENT_ROLE_FIELDS
        ),
        "passes_owner_team_validity": (all_owner_teams_valid),
        "passes_player_id_completeness": (all_player_ids_present),
        "passes_player_team_mapping": (all_players_mapped),
        "passes_unambiguous_semantics": (all_semantics_unambiguous),
    }

    status = "complete" if all(gates.values()) else "failed"

    audit = {
        "schema_version": "1.0",
        "batch_id": batch_id,
        "status": status,
        "game_count": len(seen_game_ids),
        "validation_row_count": len(validation_rows),
        "output_path": str(output_path),
        "output_sha256": output_digest,
        "gates": gates,
        "event_types": semantic_summaries,
        "games": game_summaries,
    }

    _write_json_atomically(audit_path, audit)

    if status != "complete":
        failed_gates = sorted(gate for gate, passed in gates.items() if not passed)
        raise EventOwnerSemanticError(f"Event-owner semantic gates failed: {failed_gates}")

    return EventOwnerSemanticResult(
        batch_id=batch_id,
        game_count=len(seen_game_ids),
        validation_row_count=len(validation_rows),
        event_type_count=len(semantic_summaries),
        all_player_ids_present=(all_player_ids_present),
        all_players_mapped=all_players_mapped,
        all_owner_teams_valid=(all_owner_teams_valid),
        all_semantics_unambiguous=(all_semantics_unambiguous),
        output_path=str(output_path),
        output_sha256=output_digest,
        audit_path=str(audit_path),
        already_present=already_present,
        status=status,
    )


def result_as_dict(
    result: EventOwnerSemanticResult,
) -> dict[str, Any]:
    """Convert a semantic result into a mapping."""
    return asdict(result)
