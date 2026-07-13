"""Build the first five-game NHL pilot."""

from __future__ import annotations

import json
from pathlib import Path

from nhl_ml.pilot import (
    PilotBuildError,
    build_five_game_pilot,
    result_as_dict,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    try:
        result = build_five_game_pilot(
            canonical_root=(PROJECT_ROOT / "data/interim/nhl_web"),
            raw_root=PROJECT_ROOT / "storage/raw",
            output_root=PROJECT_ROOT / "data/interim/pilot",
            audit_path=(PROJECT_ROOT / "storage/audits/local/five_game_pilot.json"),
        )
    except (FileNotFoundError, PilotBuildError) as exc:
        print(f"Pilot build failed: {exc}")
        return 1

    print(
        json.dumps(
            result_as_dict(result),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
