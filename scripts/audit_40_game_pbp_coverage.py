"""Audit local download coverage for the forty-game NHL PBP pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nhl_ml.config import ConfigError
from nhl_ml.pbp_batch import PbpBatchConfig
from nhl_ml.pbp_coverage import (
    PbpCoverageError,
    audit_pbp_download_coverage,
    result_as_dict,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Audit downloaded PBP JSON coverage for the forty-game pilot.")
    )
    parser.add_argument(
        "--downloads-dir",
        type=Path,
        default=Path.home() / "Downloads/nhl_pbp_40",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    audit_path = PROJECT_ROOT / "storage/audits/local/forty_game_pbp_coverage.json"

    try:
        config = PbpBatchConfig.from_path(PROJECT_ROOT / "configs/pbp_40_game_pilot.yaml")

        result = audit_pbp_download_coverage(
            config=config,
            downloads_dir=args.downloads_dir,
            audit_path=audit_path,
            require_complete=args.require_complete,
        )
    except (
        ConfigError,
        PbpCoverageError,
    ) as exc:
        print(f"PBP coverage audit failed: {exc}")
        return 1

    print(json.dumps(result_as_dict(result), indent=2, sort_keys=True))

    audit = json.loads(audit_path.read_text(encoding="utf-8"))

    if audit["missing_games"]:
        print("\nMissing download URLs:")

        for game in audit["missing_games"]:
            print(
                game["game_id"],
                game["download_url"],
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
