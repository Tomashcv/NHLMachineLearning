from pathlib import Path

from nhl_ml.config import ProjectConfig, SourceRegistry

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_project_configuration_loads() -> None:
    config = ProjectConfig.from_path(PROJECT_ROOT / "configs/project.yaml")

    assert config.project_name == "NHLMachineLearning"
    assert config.canonical_timezone == "UTC"
    assert config.primary_market == "full_game_moneyline"
    assert config.early_cutoff_minutes == 360
    assert config.pregame_cutoff_minutes == 60
    assert config.pilot_target_games == 40
    assert config.pilot_minimum_seasons == 5


def test_source_registry_has_unique_source_ids() -> None:
    registry = SourceRegistry.from_path(PROJECT_ROOT / "configs/source_registry.yaml")

    source_ids = [entry.source_id for entry in registry.entries]

    assert source_ids
    assert len(source_ids) == len(set(source_ids))


def test_betfair_is_research_only_and_private() -> None:
    registry = SourceRegistry.from_path(PROJECT_ROOT / "configs/source_registry.yaml")

    betfair = registry.by_id("betfair_historical")

    assert betfair.research_only is True
    assert betfair.redistribution_status == "do_not_redistribute"
    assert betfair.automation_status == "manual_download_only"
