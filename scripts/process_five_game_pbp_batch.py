"""Process the configured five-game NHL PBP pilot offline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nhl_ml.config import ConfigError
from nhl_ml.pbp_batch import (
    PbpBatchConfig,
    PbpBatchError,
    result_as_dict,
    run_pbp_batch,
)
from nhl_ml.pbp_canonical import PbpCanonicalizationError
from nhl_ml.pbp_raw import PbpRawError

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Process the five-game NHL PBP pilot from manually saved JSON files.")
    )
    parser.add_argument(
        "--downloads-dir",
        type=Path,
        default=Path.home() / "Downloads/nhl_pbp_pilot",
        help="Directory containing the five manually saved files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        config = PbpBatchConfig.from_path(PROJECT_ROOT / "configs/pbp_five_game_pilot.yaml")

        result = run_pbp_batch(
            config=config,
            downloads_dir=args.downloads_dir,
            raw_root=PROJECT_ROOT / "storage/raw",
            manifest_path=(PROJECT_ROOT / "storage/manifests/local/nhl_web_imports.jsonl"),
            canonical_output_root=(PROJECT_ROOT / "data/interim/pbp"),
            per_game_audit_root=(PROJECT_ROOT / "storage/audits/local"),
            aggregate_audit_path=(PROJECT_ROOT / "storage/audits/local/five_game_pbp_batch.json"),
        )
    except (
        ConfigError,
        FileNotFoundError,
        PbpBatchError,
        PbpCanonicalizationError,
        PbpRawError,
    ) as exc:
        print(f"Five-game PBP batch failed: {exc}")
        return 1

    print(json.dumps(result_as_dict(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
