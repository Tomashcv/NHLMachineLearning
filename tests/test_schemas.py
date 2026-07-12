from datetime import UTC, datetime, timedelta

import pytest

from nhl_ml.schemas import (
    CanonicalGame,
    CanonicalMarket,
    GameStatus,
    GameType,
    MarketQuote,
    TemporalFeatureValue,
)


def utc_time(hour: int, minute: int = 0) -> datetime:
    return datetime(2025, 1, 10, hour, minute, tzinfo=UTC)


def test_valid_final_shootout_game() -> None:
    game = CanonicalGame(
        game_id="2024020001",
        season_id="20242025",
        game_type=GameType.REGULAR_SEASON,
        scheduled_start_utc=utc_time(19),
        home_team_id="BOS",
        away_team_id="TOR",
        status=GameStatus.FINAL,
        home_score=3,
        away_score=2,
        went_to_overtime=True,
        went_to_shootout=True,
        source="pilot_fixture",
        source_observed_at_utc=utc_time(22),
        ingested_at_utc=utc_time(22, 5),
    )

    assert game.went_to_shootout is True
    assert game.home_score == 3


def test_shootout_requires_overtime() -> None:
    with pytest.raises(ValueError, match="Shootout"):
        CanonicalGame(
            game_id="2024020002",
            season_id="20242025",
            game_type=GameType.REGULAR_SEASON,
            scheduled_start_utc=utc_time(19),
            home_team_id="BOS",
            away_team_id="TOR",
            status=GameStatus.FINAL,
            home_score=3,
            away_score=2,
            went_to_overtime=False,
            went_to_shootout=True,
            source="pilot_fixture",
            source_observed_at_utc=utc_time(22),
            ingested_at_utc=utc_time(22, 5),
        )


def test_future_information_is_rejected() -> None:
    decision_time = utc_time(18)

    with pytest.raises(ValueError, match="after the model decision"):
        TemporalFeatureValue(
            game_id="2024020003",
            feature_name="confirmed_goalie_strength",
            value=0.4,
            source="pilot_fixture",
            observed_at_utc=decision_time + timedelta(minutes=5),
            decision_time_utc=decision_time,
        )


def test_timestamped_quote_is_bet_time_safe() -> None:
    quote = MarketQuote(
        game_id="2024020004",
        bookmaker="example_book",
        market=CanonicalMarket.FULL_GAME_MONEYLINE,
        outcome="home",
        decimal_odds=1.91,
        snapshot_utc=utc_time(17, 55),
        provider_updated_at_utc=utc_time(17, 54),
        ingested_at_utc=utc_time(18, 5),
    )

    assert quote.is_bet_time_safe(utc_time(18)) is True
    assert quote.is_bet_time_safe(utc_time(17, 50)) is False


def test_quote_without_provider_timestamp_is_not_bet_time_safe() -> None:
    quote = MarketQuote(
        game_id="2024020005",
        bookmaker="unknown_timestamp_book",
        market=CanonicalMarket.FULL_GAME_TOTAL,
        outcome="over",
        decimal_odds=1.95,
        line=6.0,
        snapshot_utc=utc_time(17, 55),
        provider_updated_at_utc=None,
        ingested_at_utc=utc_time(18, 5),
    )

    assert quote.is_bet_time_safe(utc_time(18)) is False
