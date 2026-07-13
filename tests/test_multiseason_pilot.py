import json
from pathlib import Path

import pytest

from nhl_ml.multiseason_pilot import (
    MultiseasonPilotError,
    PilotSelectionConfig,
    select_multiseason_pilot,
)


def make_record(
    *,
    game_id: str,
    season_id: str,
    source_date: str,
    hour: int,
    game_type: str = "regular_season",
    outcome: str = "regulation",
) -> dict:
    went_to_overtime = outcome in {"overtime_only", "shootout"}
    went_to_shootout = outcome == "shootout"

    return {
        "schema_version": "1.0",
        "source_id": "nhl_web",
        "source_raw_relative_path": (f"nhl_web/manual/{source_date}/score_fixture.json"),
        "source_raw_sha256": f"hash-{source_date}",
        "game": {
            "game_id": game_id,
            "season_id": season_id,
            "game_type": game_type,
            "scheduled_start_utc": (f"{source_date}T{hour:02d}:00:00+00:00"),
            "home_team_id": f"H{game_id}",
            "away_team_id": f"A{game_id}",
            "status": "final",
            "home_score": 3,
            "away_score": 2,
            "went_to_overtime": went_to_overtime,
            "went_to_shootout": went_to_shootout,
            "source": "nhl_web",
            "source_observed_at_utc": (f"{source_date}T23:00:00+00:00"),
            "ingested_at_utc": f"{source_date}T23:00:00+00:00",
        },
    }


def prepare_records(tmp_path: Path) -> Path:
    canonical_root = tmp_path / "data/interim/nhl_web"

    seasons = [
        "20202021",
        "20212022",
        "20222023",
        "20232024",
        "20242025",
    ]

    for season_index, season_id in enumerate(seasons, start=1):
        source_date = f"202{season_index}-01-10"
        directory = canonical_root / source_date
        directory.mkdir(parents=True)

        records = []

        for game_index in range(3):
            outcome = "regulation"
            game_type = "regular_season"

            if season_index == 1 and game_index == 0:
                outcome = "overtime_only"

            if season_index == 4 and game_index < 2:
                game_type = "playoff"

            if season_index == 5 and game_index == 0:
                outcome = "shootout"

            records.append(
                make_record(
                    game_id=f"{season_index}{game_index}",
                    season_id=season_id,
                    source_date=source_date,
                    hour=10 + game_index,
                    game_type=game_type,
                    outcome=outcome,
                )
            )

        path = directory / "games_fixture.jsonl"
        path.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
            encoding="utf-8",
        )

    return canonical_root


def test_selects_exact_balanced_pilot(tmp_path: Path) -> None:
    canonical_root = prepare_records(tmp_path)

    config = PilotSelectionConfig(
        selection_id="test_10",
        target_games=10,
        maximum_games_per_source_date=3,
        minimum_playoff_games=2,
        required_outcomes=(
            "regulation",
            "overtime_only",
            "shootout",
        ),
        season_quotas={
            "20202021": 2,
            "20212022": 2,
            "20222023": 2,
            "20232024": 2,
            "20242025": 2,
        },
    )
    config.validate()

    result = select_multiseason_pilot(
        canonical_root=canonical_root,
        config=config,
        output_path=tmp_path / "pilot/games.jsonl",
        audit_path=tmp_path / "audits/selection.json",
    )

    audit = json.loads(Path(result.audit_path).read_text(encoding="utf-8"))

    assert result.selected_game_count == 10
    assert result.playoff_count == 2
    assert result.overtime_only_count >= 1
    assert result.shootout_count >= 1
    assert audit["passes_target_count"] is True
    assert audit["passes_season_quotas"] is True
    assert audit["passes_source_date_cap"] is True
    assert audit["passes_required_outcomes"] is True


def test_selection_is_deterministic(tmp_path: Path) -> None:
    canonical_root = prepare_records(tmp_path)

    config = PilotSelectionConfig(
        selection_id="test_10",
        target_games=10,
        maximum_games_per_source_date=3,
        minimum_playoff_games=2,
        required_outcomes=(
            "regulation",
            "overtime_only",
            "shootout",
        ),
        season_quotas={
            "20202021": 2,
            "20212022": 2,
            "20222023": 2,
            "20232024": 2,
            "20242025": 2,
        },
    )

    output_path = tmp_path / "pilot/games.jsonl"
    audit_path = tmp_path / "audits/selection.json"

    first = select_multiseason_pilot(
        canonical_root=canonical_root,
        config=config,
        output_path=output_path,
        audit_path=audit_path,
    )
    first_bytes = output_path.read_bytes()

    second = select_multiseason_pilot(
        canonical_root=canonical_root,
        config=config,
        output_path=output_path,
        audit_path=audit_path,
    )

    assert output_path.read_bytes() == first_bytes
    assert first.selection_sha256 == second.selection_sha256


def test_impossible_source_date_cap_is_rejected(
    tmp_path: Path,
) -> None:
    canonical_root = prepare_records(tmp_path)

    config = PilotSelectionConfig(
        selection_id="impossible",
        target_games=15,
        maximum_games_per_source_date=2,
        minimum_playoff_games=2,
        required_outcomes=(
            "regulation",
            "overtime_only",
            "shootout",
        ),
        season_quotas={
            "20202021": 3,
            "20212022": 3,
            "20222023": 3,
            "20232024": 3,
            "20242025": 3,
        },
    )

    with pytest.raises(
        MultiseasonPilotError,
        match="Could not satisfy quota",
    ):
        select_multiseason_pilot(
            canonical_root=canonical_root,
            config=config,
            output_path=tmp_path / "pilot/games.jsonl",
            audit_path=tmp_path / "audits/selection.json",
        )
