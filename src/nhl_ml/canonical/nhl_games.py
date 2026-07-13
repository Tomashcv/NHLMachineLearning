"""Canonical NHL game parsing from manually imported daily-score payloads."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nhl_ml.schemas import CanonicalGame, GameStatus, GameType


class NHLCanonicalizationError(ValueError):
    """Raised when an NHL payload cannot be canonicalized safely."""


@dataclass(frozen=True, slots=True)
class CanonicalizationResult:
    """Summary of one deterministic canonicalization run."""

    source_id: str
    requested_date: str
    raw_relative_path: str
    raw_sha256: str
    output_relative_path: str
    output_sha256: str
    game_count: int
    final_count: int
    regulation_count: int
    overtime_only_count: int
    shootout_count: int
    already_present: bool


_GAME_TYPE_MAP = {
    1: GameType.PRESEASON,
    2: GameType.REGULAR_SEASON,
    3: GameType.PLAYOFF,
    4: GameType.ALL_STAR,
}

_GAME_STATUS_MAP = {
    "FUT": GameStatus.SCHEDULED,
    "PRE": GameStatus.SCHEDULED,
    "LIVE": GameStatus.LIVE,
    "CRIT": GameStatus.LIVE,
    "OFF": GameStatus.FINAL,
    "FINAL": GameStatus.FINAL,
    "PPD": GameStatus.POSTPONED,
    "POSTPONED": GameStatus.POSTPONED,
    "SUSP": GameStatus.SUSPENDED,
    "SUSPENDED": GameStatus.SUSPENDED,
    "CANC": GameStatus.CANCELLED,
    "CANCELLED": GameStatus.CANCELLED,
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _require_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise NHLCanonicalizationError(f"{field_name} must be a JSON object")

    return value


def _parse_utc_datetime(value: Any, field_name: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise NHLCanonicalizationError(f"{field_name} must be a non-empty ISO timestamp")

    normalized = value.strip().replace("Z", "+00:00")

    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise NHLCanonicalizationError(
            f"{field_name} is not a valid ISO timestamp: {value!r}"
        ) from exc

    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise NHLCanonicalizationError(f"{field_name} must be timezone-aware")

    if parsed.utcoffset().total_seconds() != 0:
        raise NHLCanonicalizationError(f"{field_name} must be represented in UTC")

    return parsed.astimezone(UTC)


def _parse_game_type(value: Any) -> GameType:
    try:
        game_type_id = int(value)
    except (TypeError, ValueError) as exc:
        raise NHLCanonicalizationError(f"Invalid NHL game type: {value!r}") from exc

    try:
        return _GAME_TYPE_MAP[game_type_id]
    except KeyError as exc:
        raise NHLCanonicalizationError(f"Unsupported NHL game type: {game_type_id}") from exc


def _parse_game_status(value: Any) -> GameStatus:
    if not isinstance(value, str):
        raise NHLCanonicalizationError(f"Invalid NHL game state: {value!r}")

    normalized = value.strip().upper()

    try:
        return _GAME_STATUS_MAP[normalized]
    except KeyError as exc:
        raise NHLCanonicalizationError(f"Unsupported NHL game state: {normalized!r}") from exc


def _parse_team_id(team: dict[str, Any], field_name: str) -> str:
    team_id = team.get("id")

    if team_id is None:
        raise NHLCanonicalizationError(f"{field_name}.id is required")

    parsed = str(team_id).strip()

    if not parsed:
        raise NHLCanonicalizationError(f"{field_name}.id cannot be empty")

    return parsed


def _parse_score(team: dict[str, Any], field_name: str) -> int:
    score = team.get("score")

    try:
        parsed = int(score)
    except (TypeError, ValueError) as exc:
        raise NHLCanonicalizationError(f"{field_name}.score must be an integer") from exc

    if parsed < 0:
        raise NHLCanonicalizationError(f"{field_name}.score cannot be negative")

    return parsed


def _parse_final_outcome(game: dict[str, Any]) -> tuple[bool, bool]:
    outcome = _require_mapping(
        game.get("gameOutcome"),
        "gameOutcome",
    )

    period_type = outcome.get("lastPeriodType")

    if not isinstance(period_type, str):
        raise NHLCanonicalizationError("gameOutcome.lastPeriodType is required for final games")

    normalized = period_type.strip().upper()

    if normalized == "REG":
        return False, False

    if normalized == "OT":
        return True, False

    if normalized == "SO":
        return True, True

    raise NHLCanonicalizationError(f"Unsupported final period type: {normalized!r}")


def parse_score_payload(
    *,
    payload: dict[str, Any],
    source_observed_at_utc: datetime,
    ingested_at_utc: datetime,
) -> tuple[CanonicalGame, ...]:
    """Parse one NHL daily-score payload into canonical games."""
    games_raw = payload.get("games")

    if not isinstance(games_raw, list):
        raise NHLCanonicalizationError("Expected the score payload to contain a 'games' list")

    canonical_games: list[CanonicalGame] = []
    seen_game_ids: set[str] = set()

    for index, game_raw in enumerate(games_raw):
        game = _require_mapping(game_raw, f"games[{index}]")

        game_id_raw = game.get("id")
        if game_id_raw is None:
            raise NHLCanonicalizationError(f"games[{index}].id is required")

        game_id = str(game_id_raw).strip()
        if not game_id:
            raise NHLCanonicalizationError(f"games[{index}].id cannot be empty")

        if game_id in seen_game_ids:
            raise NHLCanonicalizationError(f"Duplicate game ID in payload: {game_id}")

        seen_game_ids.add(game_id)

        season_raw = game.get("season")
        if season_raw is None:
            raise NHLCanonicalizationError(f"games[{index}].season is required")

        season_id = str(season_raw).strip()

        home_team = _require_mapping(
            game.get("homeTeam"),
            f"games[{index}].homeTeam",
        )
        away_team = _require_mapping(
            game.get("awayTeam"),
            f"games[{index}].awayTeam",
        )

        status = _parse_game_status(game.get("gameState"))

        home_score: int | None = None
        away_score: int | None = None
        went_to_overtime: bool | None = None
        went_to_shootout: bool | None = None

        if status is GameStatus.FINAL:
            home_score = _parse_score(
                home_team,
                f"games[{index}].homeTeam",
            )
            away_score = _parse_score(
                away_team,
                f"games[{index}].awayTeam",
            )
            went_to_overtime, went_to_shootout = _parse_final_outcome(game)
        elif status is GameStatus.LIVE:
            if "score" in home_team and "score" in away_team:
                home_score = _parse_score(
                    home_team,
                    f"games[{index}].homeTeam",
                )
                away_score = _parse_score(
                    away_team,
                    f"games[{index}].awayTeam",
                )

        canonical_games.append(
            CanonicalGame(
                game_id=game_id,
                season_id=season_id,
                game_type=_parse_game_type(game.get("gameType")),
                scheduled_start_utc=_parse_utc_datetime(
                    game.get("startTimeUTC"),
                    f"games[{index}].startTimeUTC",
                ),
                home_team_id=_parse_team_id(
                    home_team,
                    f"games[{index}].homeTeam",
                ),
                away_team_id=_parse_team_id(
                    away_team,
                    f"games[{index}].awayTeam",
                ),
                status=status,
                home_score=home_score,
                away_score=away_score,
                went_to_overtime=went_to_overtime,
                went_to_shootout=went_to_shootout,
                source="nhl_web",
                source_observed_at_utc=source_observed_at_utc,
                ingested_at_utc=ingested_at_utc,
            )
        )

    return tuple(
        sorted(
            canonical_games,
            key=lambda game: (
                game.scheduled_start_utc,
                game.game_id,
            ),
        )
    )


def _load_json_mapping(data: bytes, path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NHLCanonicalizationError(f"Raw file is not valid UTF-8 JSON: {path}") from exc

    return _require_mapping(payload, "raw payload")


def _find_import_record(
    *,
    manifest_path: Path,
    raw_relative_path: str,
    raw_sha256: str,
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
                raise NHLCanonicalizationError(
                    f"Invalid JSONL manifest record at line {line_number}"
                ) from exc

            if not isinstance(record, dict):
                continue

            if (
                record.get("raw_relative_path") == raw_relative_path
                and record.get("sha256") == raw_sha256
            ):
                return record

    raise NHLCanonicalizationError("No matching raw import record was found in the manifest")


def _game_as_mapping(game: CanonicalGame) -> dict[str, Any]:
    return {
        "game_id": game.game_id,
        "season_id": game.season_id,
        "game_type": game.game_type.value,
        "scheduled_start_utc": game.scheduled_start_utc.isoformat(),
        "home_team_id": game.home_team_id,
        "away_team_id": game.away_team_id,
        "status": game.status.value,
        "home_score": game.home_score,
        "away_score": game.away_score,
        "went_to_overtime": game.went_to_overtime,
        "went_to_shootout": game.went_to_shootout,
        "source": game.source,
        "source_observed_at_utc": (game.source_observed_at_utc.isoformat()),
        "ingested_at_utc": game.ingested_at_utc.isoformat(),
    }


def _serialize_games(
    *,
    games: tuple[CanonicalGame, ...],
    raw_relative_path: str,
    raw_sha256: str,
) -> bytes:
    lines: list[str] = []

    for game in games:
        record = {
            "schema_version": "1.0",
            "source_id": "nhl_web",
            "source_raw_relative_path": raw_relative_path,
            "source_raw_sha256": raw_sha256,
            "temporal_classification": "postgame_result_data",
            "source_observed_at_basis": "manual_import_timestamp_proxy",
            "game": _game_as_mapping(game),
        }

        lines.append(
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )

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


def canonicalize_score_file(
    *,
    raw_file: Path,
    raw_root: Path,
    import_manifest_path: Path,
    output_root: Path,
) -> CanonicalizationResult:
    """Create deterministic canonical game JSONL from one raw score file."""
    raw_root = raw_root.expanduser().resolve()
    raw_file = raw_file.expanduser().resolve()

    if not raw_file.is_file():
        raise FileNotFoundError(f"Raw file does not exist: {raw_file}")

    try:
        raw_relative_path = raw_file.relative_to(raw_root).as_posix()
    except ValueError as exc:
        raise NHLCanonicalizationError(
            f"Raw file is outside the configured raw root: {raw_file}"
        ) from exc

    raw_data = raw_file.read_bytes()
    raw_digest = _sha256(raw_data)
    payload = _load_json_mapping(raw_data, raw_file)

    import_record = _find_import_record(
        manifest_path=import_manifest_path,
        raw_relative_path=raw_relative_path,
        raw_sha256=raw_digest,
    )

    requested_date = str(import_record.get("requested_date", "")).strip()
    if not requested_date:
        raise NHLCanonicalizationError("Import manifest record is missing requested_date")

    imported_at_raw = import_record.get("imported_at_utc")
    imported_at = _parse_utc_datetime(
        imported_at_raw,
        "manifest.imported_at_utc",
    )

    games = parse_score_payload(
        payload=payload,
        source_observed_at_utc=imported_at,
        ingested_at_utc=imported_at,
    )

    output_data = _serialize_games(
        games=games,
        raw_relative_path=raw_relative_path,
        raw_sha256=raw_digest,
    )
    output_digest = _sha256(output_data)

    output_relative_path = Path("nhl_web") / requested_date / f"games_{raw_digest[:12]}.jsonl"
    destination = output_root / output_relative_path

    already_present = destination.is_file()

    if already_present:
        existing_data = destination.read_bytes()

        if existing_data != output_data:
            raise NHLCanonicalizationError(
                f"Existing canonical file is not deterministic: {destination}"
            )
    else:
        _write_bytes_atomically(destination, output_data)

    final_games = [game for game in games if game.status is GameStatus.FINAL]

    regulation_count = sum(game.went_to_overtime is False for game in final_games)
    overtime_only_count = sum(
        game.went_to_overtime is True and game.went_to_shootout is False for game in final_games
    )
    shootout_count = sum(game.went_to_shootout is True for game in final_games)

    return CanonicalizationResult(
        source_id="nhl_web",
        requested_date=requested_date,
        raw_relative_path=raw_relative_path,
        raw_sha256=raw_digest,
        output_relative_path=output_relative_path.as_posix(),
        output_sha256=output_digest,
        game_count=len(games),
        final_count=len(final_games),
        regulation_count=regulation_count,
        overtime_only_count=overtime_only_count,
        shootout_count=shootout_count,
        already_present=already_present,
    )


def result_as_dict(
    result: CanonicalizationResult,
) -> dict[str, Any]:
    """Convert a canonicalization result to a serializable mapping."""
    return asdict(result)
