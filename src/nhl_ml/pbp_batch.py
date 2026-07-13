"""Offline batch processing and reconciliation of NHL play-by-play files."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from nhl_ml.config import ConfigError, load_yaml
from nhl_ml.pbp_canonical import canonicalize_pbp_file
from nhl_ml.pbp_raw import process_manual_pbp_file

_ALLOWED_OUTCOMES = {
    "regulation",
    "overtime_only",
    "shootout",
}


class PbpBatchError(ValueError):
    """Raised when a PBP batch fails validation or reconciliation."""


@dataclass(frozen=True, slots=True)
class PbpBatchGame:
    """One expected game in the offline PBP batch."""

    game_id: str
    expected_outcome: str
    source_filename: str

    def validate(self) -> None:
        if not self.game_id.strip():
            raise ConfigError("PBP batch game_id cannot be empty")

        if self.expected_outcome not in _ALLOWED_OUTCOMES:
            raise ConfigError(f"Unsupported expected outcome: {self.expected_outcome}")

        if not self.source_filename.strip():
            raise ConfigError("PBP source_filename cannot be empty")


@dataclass(frozen=True, slots=True)
class PbpBatchConfig:
    """Frozen configuration for an offline PBP batch."""

    batch_id: str
    expected_game_count: int
    games: tuple[PbpBatchGame, ...]

    @classmethod
    def from_path(cls, path: Path) -> PbpBatchConfig:
        payload = load_yaml(path)

        try:
            batch = payload["batch"]
            raw_games = payload["games"]

            if not isinstance(raw_games, list):
                raise ConfigError("games must be a YAML list")

            games = tuple(
                PbpBatchGame(
                    game_id=str(raw["game_id"]).strip(),
                    expected_outcome=str(raw["expected_outcome"]).strip(),
                    source_filename=str(raw["source_filename"]).strip(),
                )
                for raw in raw_games
            )

            config = cls(
                batch_id=str(batch["batch_id"]).strip(),
                expected_game_count=int(batch["expected_game_count"]),
                games=games,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigError(f"Invalid PBP batch configuration: {exc}") from exc

        config.validate()
        return config

    def validate(self) -> None:
        if not self.batch_id:
            raise ConfigError("PBP batch_id cannot be empty")

        if self.expected_game_count <= 0:
            raise ConfigError("expected_game_count must be positive")

        if len(self.games) != self.expected_game_count:
            raise ConfigError("Configured games do not match expected_game_count")

        game_ids = [game.game_id for game in self.games]

        if len(game_ids) != len(set(game_ids)):
            raise ConfigError("PBP batch contains duplicate game IDs")

        filenames = [game.source_filename for game in self.games]

        if len(filenames) != len(set(filenames)):
            raise ConfigError("PBP batch contains duplicate source filenames")

        for game in self.games:
            game.validate()


@dataclass(frozen=True, slots=True)
class PbpBatchResult:
    """Summary of a completed five-game PBP batch."""

    batch_id: str
    processed_game_count: int
    regulation_count: int
    overtime_only_count: int
    shootout_count: int
    all_sog_reconciled: bool
    all_applicable_scores_reconciled: bool
    all_outcomes_matched: bool
    all_core_events_have_team: bool
    batch_sha256: str
    audit_path: str
    status: str


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PbpBatchError(f"Invalid JSON file: {path}") from exc

    if not isinstance(payload, dict):
        raise PbpBatchError(f"Expected JSON object in {path}")

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
                raise PbpBatchError(f"Invalid JSONL in {path} at line {line_number}") from exc

            if not isinstance(record, dict):
                raise PbpBatchError(f"Expected JSON object in {path} at line {line_number}")

            records.append(record)

    return records


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


def _observed_outcome(
    canonical_records: list[dict[str, Any]],
) -> str:
    period_types = {
        str(record["event"].get("period_type", "")).upper() for record in canonical_records
    }

    if "SO" in period_types:
        return "shootout"

    if "OT" in period_types:
        return "overtime_only"

    return "regulation"


def run_pbp_batch(
    *,
    config: PbpBatchConfig,
    downloads_dir: Path,
    raw_root: Path,
    manifest_path: Path,
    canonical_output_root: Path,
    per_game_audit_root: Path,
    aggregate_audit_path: Path,
) -> PbpBatchResult:
    """Process, canonicalize and reconcile a configured PBP batch."""
    downloads_dir = downloads_dir.expanduser().resolve()

    if not downloads_dir.is_dir():
        raise FileNotFoundError(f"Downloads directory does not exist: {downloads_dir}")

    game_summaries: list[dict[str, Any]] = []
    outcome_counts: Counter[str] = Counter()

    for configured_game in config.games:
        input_path = downloads_dir / configured_game.source_filename

        if not input_path.is_file():
            raise FileNotFoundError(f"Missing configured PBP file: {input_path}")

        raw_audit_path = per_game_audit_root / f"pbp_{configured_game.game_id}.json"

        raw_result = process_manual_pbp_file(
            input_path=input_path,
            expected_game_id=configured_game.game_id,
            raw_root=raw_root,
            manifest_path=manifest_path,
            audit_path=raw_audit_path,
        )

        raw_file = raw_root / raw_result.raw_relative_path
        canonical_audit_path = per_game_audit_root / f"pbp_canonical_{configured_game.game_id}.json"

        canonical_result = canonicalize_pbp_file(
            raw_file=raw_file,
            raw_root=raw_root,
            import_manifest_path=manifest_path,
            output_root=canonical_output_root,
            audit_path=canonical_audit_path,
        )

        canonical_file = canonical_output_root / canonical_result.output_relative_path
        canonical_records = _read_jsonl(canonical_file)
        canonical_audit = _read_json(canonical_audit_path)

        observed_outcome = _observed_outcome(canonical_records)
        outcome_counts[observed_outcome] += 1

        outcome_matches = observed_outcome == configured_game.expected_outcome

        score_status = canonical_audit["score_reconciliation_passed"]

        if observed_outcome == "shootout":
            applicable_score_reconciliation_passed = score_status is None
        else:
            applicable_score_reconciliation_passed = score_status is True

        game_summaries.append(
            {
                "game_id": configured_game.game_id,
                "source_filename": (configured_game.source_filename),
                "expected_outcome": (configured_game.expected_outcome),
                "observed_outcome": observed_outcome,
                "outcome_matches": outcome_matches,
                "raw_sha256": raw_result.raw_sha256,
                "canonical_sha256": (canonical_result.output_sha256),
                "raw_play_count": raw_result.play_count,
                "canonical_event_count": (canonical_result.event_count),
                "event_counts_match": (raw_result.play_count == canonical_result.event_count),
                "score_reconciliation_status": score_status,
                "applicable_score_reconciliation_passed": (applicable_score_reconciliation_passed),
                "sog_reconciliation_passed": (canonical_result.sog_reconciliation_passed),
                "missing_owner_team_core_event_count": (
                    canonical_audit["missing_owner_team_core_event_count"]
                ),
                "empty_net_goal_candidate_count": (canonical_result.empty_net_goal_candidate_count),
                "shot_coordinate_coverage": (canonical_audit["shot_coordinate_coverage"]),
                "unmapped_detail_key_counts": (canonical_audit["unmapped_detail_key_counts"]),
            }
        )

    all_sog_reconciled = all(game["sog_reconciliation_passed"] for game in game_summaries)
    all_applicable_scores_reconciled = all(
        game["applicable_score_reconciliation_passed"] for game in game_summaries
    )
    all_outcomes_matched = all(game["outcome_matches"] for game in game_summaries)
    all_event_counts_match = all(game["event_counts_match"] for game in game_summaries)
    all_core_events_have_team = all(
        game["missing_owner_team_core_event_count"] == 0 for game in game_summaries
    )

    required_outcomes_present = all(outcome_counts[outcome] > 0 for outcome in _ALLOWED_OUTCOMES)

    processed_game_count = len(game_summaries)

    batch_material = "\n".join(
        f"{game['game_id']}:{game['canonical_sha256']}" for game in game_summaries
    ).encode("utf-8")
    batch_digest = _sha256(batch_material)

    gates = {
        "passes_expected_game_count": (processed_game_count == config.expected_game_count),
        "passes_unique_game_ids": (
            len({game["game_id"] for game in game_summaries}) == processed_game_count
        ),
        "passes_event_count_reconciliation": (all_event_counts_match),
        "passes_sog_reconciliation": all_sog_reconciled,
        "passes_applicable_score_reconciliation": (all_applicable_scores_reconciled),
        "passes_expected_outcomes": all_outcomes_matched,
        "passes_required_outcome_coverage": (required_outcomes_present),
        "passes_core_team_ownership": (all_core_events_have_team),
    }

    status = "complete" if all(gates.values()) else "failed"

    aggregate_audit = {
        "schema_version": "1.0",
        "batch_id": config.batch_id,
        "status": status,
        "processed_game_count": processed_game_count,
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "batch_sha256": batch_digest,
        "gates": gates,
        "games": game_summaries,
    }

    _write_json_atomically(
        aggregate_audit_path,
        aggregate_audit,
    )

    if status != "complete":
        failed_gates = sorted(gate for gate, passed in gates.items() if not passed)
        raise PbpBatchError(f"PBP batch failed gates: {failed_gates}")

    return PbpBatchResult(
        batch_id=config.batch_id,
        processed_game_count=processed_game_count,
        regulation_count=outcome_counts["regulation"],
        overtime_only_count=outcome_counts["overtime_only"],
        shootout_count=outcome_counts["shootout"],
        all_sog_reconciled=all_sog_reconciled,
        all_applicable_scores_reconciled=(all_applicable_scores_reconciled),
        all_outcomes_matched=all_outcomes_matched,
        all_core_events_have_team=all_core_events_have_team,
        batch_sha256=batch_digest,
        audit_path=str(aggregate_audit_path),
        status=status,
    )


def result_as_dict(result: PbpBatchResult) -> dict[str, Any]:
    """Convert a PBP batch result into a mapping."""
    return asdict(result)
