"""Build the frozen forty-game NHL PBP manifest."""

from __future__ import annotations

import json
from pathlib import Path

from nhl_ml.pbp_manifest import (
    PbpManifestError,
    build_pbp_manifest,
    result_as_dict,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    try:
        result = build_pbp_manifest(
            pilot_path=(PROJECT_ROOT / "data/interim/pilot/multiseason_40_games.jsonl"),
            selection_audit_path=(
                PROJECT_ROOT / "storage/audits/local/multiseason_40_selection.json"
            ),
            output_path=(PROJECT_ROOT / "configs/pbp_40_game_pilot.yaml"),
            batch_id="multiseason_40_pbp_v1",
            expected_game_count=40,
        )
    except (
        FileNotFoundError,
        PbpManifestError,
    ) as exc:
        print(f"PBP manifest build failed: {exc}")
        return 1

    print(json.dumps(result_as_dict(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
