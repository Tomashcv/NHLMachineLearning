"""Build the frozen four-season regular-season target manifest."""

from __future__ import annotations

import json
from pathlib import Path

from nhl_ml.season_scale_manifest import (
    SeasonScaleManifestError,
    build_season_scale_manifest,
    result_as_dict,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    try:
        result = build_season_scale_manifest(
            config_path=(PROJECT_ROOT / "configs/season_scale_regular_season.yaml"),
            manifest_path=(
                PROJECT_ROOT / "data/interim/manifests/season_scale_regular_season.jsonl"
            ),
            audit_path=(
                PROJECT_ROOT / "storage/audits/local/season_scale_regular_season_manifest.json"
            ),
        )
    except (
        FileNotFoundError,
        SeasonScaleManifestError,
    ) as exc:
        print(f"Season-scale manifest build failed: {exc}")
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
