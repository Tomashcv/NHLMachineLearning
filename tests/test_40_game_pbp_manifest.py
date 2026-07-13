from collections import Counter
from pathlib import Path

from nhl_ml.pbp_batch import PbpBatchConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/pbp_40_game_pilot.yaml"


def test_real_forty_game_pbp_manifest_is_frozen() -> None:
    config = PbpBatchConfig.from_path(CONFIG_PATH)

    assert config.batch_id == "multiseason_40_pbp_v1"
    assert config.expected_game_count == 40
    assert len(config.games) == 40

    game_ids = [game.game_id for game in config.games]
    filenames = [game.source_filename for game in config.games]
    outcomes = Counter(game.expected_outcome for game in config.games)

    assert len(set(game_ids)) == 40
    assert len(set(filenames)) == 40
    assert outcomes == {
        "regulation": 35,
        "overtime_only": 4,
        "shootout": 1,
    }

    for game in config.games:
        assert game.source_filename == f"nhl_pbp_{game.game_id}.json"
