"""Build separate audited NHL experiment split files."""

from __future__ import annotations

import json
from pathlib import Path

from nhl_ml.experiment_dataset import (
    ExperimentDatasetError,
    build_experiment_dataset_bundle,
    result_as_dict,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    try:
        result = build_experiment_dataset_bundle(
            source_path=(
                PROJECT_ROOT / "data/interim/model_ready/"
                "season_scale/"
                "nhl_model_ready_"
                "2021_2025.jsonl"
            ),
            source_audit_path=(
                PROJECT_ROOT / "storage/audits/local/season_scale/nhl_model_ready_2021_2025.json"
            ),
            output_dir=(PROJECT_ROOT / "data/interim/experiment/season_scale"),
            audit_path=(
                PROJECT_ROOT / "storage/audits/local/"
                "season_scale/"
                "experiment_dataset_bundle_"
                "2021_2025.json"
            ),
        )
    except (
        ExperimentDatasetError,
        FileNotFoundError,
    ) as exc:
        print(f"Experiment bundle build failed: {exc}")
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
