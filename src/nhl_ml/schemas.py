"""Canonical schemas shared across ingestion, features and market evaluation."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum


class GameType(StrEnum):
    """Supported NHL game types."""

    PRESEASON = "preseason"
    REGULAR_SEASON = "regular_season"
    PLAYOFF = "playoff"
    ALL_STAR = "all_star"


class GameStatus(StrEnum):
    """Canonical lifecycle states for NHL games."""

    SCHEDULED = "scheduled"
    LIVE = "live"
    FINAL = "final"
    POSTPONED = "postponed"
    SUSPENDED = "suspended"
    CANCELLED = "cancelled"


class CanonicalMarket(StrEnum):
    """Initial canonical NHL market definitions."""

    FULL_GAME_MONEYLINE = "full_game_moneyline"
    REGULATION_3WAY = "regulation_3way"
    PUCK_LINE = "puck_line"
    FULL_GAME_TOTAL = "full_game_total"


class MarketRole(StrEnum):
    """Documented role of one market observation."""

    OBSERVED = "observed"
    OPENING = "opening"
    CLOSING = "closing"


def require_utc(value: datetime, field_name: str) -> None:
    """Require a timezone-aware datetime represented in UTC."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")

    if value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must be represented in UTC")


@dataclass(frozen=True, slots=True)
class CanonicalGame:
    """Provider-independent NHL game record."""

    game_id: str
    season_id: str
    game_type: GameType
    scheduled_start_utc: datetime
    home_team_id: str
    away_team_id: str
    status: GameStatus
    home_score: int | None
    away_score: int | None
    went_to_overtime: bool | None
    went_to_shootout: bool | None
    source: str
    source_observed_at_utc: datetime
    ingested_at_utc: datetime

    def __post_init__(self) -> None:
        require_utc(self.scheduled_start_utc, "scheduled_start_utc")
        require_utc(
            self.source_observed_at_utc,
            "source_observed_at_utc",
        )
        require_utc(self.ingested_at_utc, "ingested_at_utc")

        if not self.game_id.strip():
            raise ValueError("game_id cannot be empty")

        if self.home_team_id == self.away_team_id:
            raise ValueError("Home and away teams must be different")

        if (self.home_score is None) != (self.away_score is None):
            raise ValueError("Home and away scores must both be present or absent")

        for score in (self.home_score, self.away_score):
            if score is not None and score < 0:
                raise ValueError("Scores cannot be negative")

        if self.status is GameStatus.FINAL:
            if self.home_score is None or self.away_score is None:
                raise ValueError("Final games must have scores")

            if self.home_score == self.away_score:
                raise ValueError("Final NHL games cannot remain tied")

        if self.went_to_shootout is True and self.went_to_overtime is not True:
            raise ValueError("Shootout games must also be overtime games")

        if self.source_observed_at_utc > self.ingested_at_utc:
            raise ValueError("Source observation cannot occur after ingestion")


@dataclass(frozen=True, slots=True)
class TemporalFeatureValue:
    """One feature value with an explicit information timestamp."""

    game_id: str
    feature_name: str
    value: float | int
    source: str
    observed_at_utc: datetime
    decision_time_utc: datetime

    def __post_init__(self) -> None:
        require_utc(self.observed_at_utc, "observed_at_utc")
        require_utc(self.decision_time_utc, "decision_time_utc")

        if not self.feature_name.strip():
            raise ValueError("feature_name cannot be empty")

        if self.observed_at_utc > self.decision_time_utc:
            raise ValueError("Feature was observed after the model decision cutoff")


@dataclass(frozen=True, slots=True)
class MarketQuote:
    """One timestamped bookmaker or exchange market observation."""

    game_id: str
    bookmaker: str
    market: CanonicalMarket
    outcome: str
    decimal_odds: float
    snapshot_utc: datetime
    provider_updated_at_utc: datetime | None
    ingested_at_utc: datetime
    line: float | None = None
    role: MarketRole = MarketRole.OBSERVED

    def __post_init__(self) -> None:
        require_utc(self.snapshot_utc, "snapshot_utc")
        require_utc(self.ingested_at_utc, "ingested_at_utc")

        if self.provider_updated_at_utc is not None:
            require_utc(
                self.provider_updated_at_utc,
                "provider_updated_at_utc",
            )

        if not self.bookmaker.strip():
            raise ValueError("bookmaker cannot be empty")

        if not self.outcome.strip():
            raise ValueError("outcome cannot be empty")

        if self.decimal_odds <= 1.0:
            raise ValueError("Decimal odds must be greater than 1.0")

        if self.snapshot_utc > self.ingested_at_utc:
            raise ValueError("Snapshot cannot occur after ingestion")

        if (
            self.provider_updated_at_utc is not None
            and self.provider_updated_at_utc > self.ingested_at_utc
        ):
            raise ValueError("Provider update cannot occur after ingestion")

    def is_bet_time_safe(self, decision_time_utc: datetime) -> bool:
        """Return whether the quote was available by the decision cutoff."""
        require_utc(decision_time_utc, "decision_time_utc")

        if self.provider_updated_at_utc is None:
            return False

        latest_information_time = max(
            self.snapshot_utc,
            self.provider_updated_at_utc,
        )
        return latest_information_time <= decision_time_utc
