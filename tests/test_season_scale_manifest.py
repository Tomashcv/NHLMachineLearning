import json
from pathlib import Path

import pytest
import yaml

from nhl_ml.season_scale_manifest import (
    SeasonScaleManifestError,
    build_season_scale_manifest,
)


def write_config(
    path: Path,
    *,
    total: int = 4,
) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0",
                "corpus": {
                    "corpus_id": "fixture_corpus",
                    "game_type_code": "02",
                    "expected_game_count": total,
                    "playoffs_included": False,
                    "split_policy": ("frozen_by_season"),
                },
                "seasons": [
                    {
                        "season_id": "20212022",
                        "start_year": 2021,
                        "first_sequence": 1,
                        "last_sequence": 2,
                        "expected_game_count": 2,
                        "split_role": "development",
                    },
                    {
                        "season_id": "20222023",
                        "start_year": 2022,
                        "first_sequence": 1,
                        "last_sequence": 2,
                        "expected_game_count": 2,
                        "split_role": "validation",
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def build_fixture(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    write_config(config_path)

    return build_season_scale_manifest(
        config_path=config_path,
        manifest_path=tmp_path / "manifest.jsonl",
        audit_path=tmp_path / "audit.json",
    )


def test_builds_exact_candidate_ids(
    tmp_path: Path,
) -> None:
    result = build_fixture(tmp_path)

    rows = [
        json.loads(line)
        for line in Path(result.manifest_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert result.status == "complete"
    assert result.game_count == 4
    assert [row["game_id"] for row in rows] == [
        "2021020001",
        "2021020002",
        "2022020001",
        "2022020002",
    ]
    assert all(row["target_status"] == "candidate_unverified" for row in rows)


def test_manifest_is_deterministic(
    tmp_path: Path,
) -> None:
    first = build_fixture(tmp_path)
    first_bytes = Path(first.manifest_path).read_bytes()

    second = build_fixture(tmp_path)

    assert second.already_present is True
    assert second.manifest_sha256 == first.manifest_sha256
    assert Path(second.manifest_path).read_bytes() == first_bytes


def test_total_count_mismatch_fails(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    write_config(config_path, total=5)

    with pytest.raises(
        SeasonScaleManifestError,
        match="gates failed",
    ):
        build_season_scale_manifest(
            config_path=config_path,
            manifest_path=(tmp_path / "manifest.jsonl"),
            audit_path=tmp_path / "audit.json",
        )
