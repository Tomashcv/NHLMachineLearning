"""Build the deterministic forty-game multiseason NHL pilot."""

from __future__ import annotations

import json
from pathlib import Path

from nhl_ml.config import ConfigError
from nhl_ml.multiseason_pilot import (
    MultiseasonPilotError,
    PilotSelectionConfig,
    result_as_dict,
    select_multiseason_pilot,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    try:
        config = PilotSelectionConfig.from_path(PROJECT_ROOT / "configs/pilot_selection.yaml")

        result = select_multiseason_pilot(
            canonical_root=PROJECT_ROOT / "data/interim/nhl_web",
            config=config,
            output_path=(PROJECT_ROOT / "data/interim/pilot/multiseason_40_games.jsonl"),
            audit_path=(PROJECT_ROOT / "storage/audits/local/multiseason_40_selection.json"),
        )
    except (
        ConfigError,
        FileNotFoundError,
        MultiseasonPilotError,
    ) as exc:
        print(f"Pilot selection failed: {exc}")
        return 1

    print(json.dumps(result_as_dict(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
