"""Run NHL moneyline baselines on the development selection season."""

from __future__ import annotations

import json
from pathlib import Path

from nhl_ml.moneyline_baselines import (
    MoneylineBaselineError,
    result_as_dict,
    run_moneyline_baseline_selection,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    try:
        result = run_moneyline_baseline_selection(
            bundle_audit_path=(
                PROJECT_ROOT / "storage/audits/local/"
                "season_scale/"
                "experiment_dataset_bundle_"
                "2021_2025.json"
            ),
            report_path=(
                PROJECT_ROOT / "storage/reports/local/moneyline/baseline_selection_20222023.json"
            ),
            expected_season_counts={
                "20212022": 1312,
                "20222023": 1312,
            },
        )
    except (
        FileNotFoundError,
        MoneylineBaselineError,
    ) as exc:
        print(f"Moneyline baseline selection failed: {exc}")
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
