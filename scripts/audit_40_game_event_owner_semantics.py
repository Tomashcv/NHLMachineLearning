"""Audit eventOwnerTeamId semantics across the forty-game NHL pilot."""

from __future__ import annotations

import json
from pathlib import Path

from nhl_ml.event_owner_semantics import (
    EventOwnerSemanticError,
    audit_event_owner_semantics,
    result_as_dict,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    try:
        result = audit_event_owner_semantics(
            pbp_manifest_path=(PROJECT_ROOT / "configs/pbp_40_game_pilot.yaml"),
            raw_root=PROJECT_ROOT / "storage/raw",
            canonical_output_root=(PROJECT_ROOT / "data/interim/pbp"),
            per_game_audit_root=(PROJECT_ROOT / "storage/audits/local"),
            output_path=(PROJECT_ROOT / "data/interim/semantics/multiseason_40_event_owner.jsonl"),
            audit_path=(PROJECT_ROOT / "storage/audits/local/multiseason_40_event_owner.json"),
        )
    except (
        FileNotFoundError,
        EventOwnerSemanticError,
    ) as exc:
        print(f"Event-owner semantic audit failed: {exc}")
        return 1

    print(json.dumps(result_as_dict(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
