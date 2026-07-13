import hashlib
import json
from pathlib import Path

from nhl_ml.canonical.nhl_games import canonicalize_score_file
from nhl_ml.pilot import build_five_game_pilot

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = PROJECT_ROOT / "tests/fixtures/nhl_score_five_game_pilot.json"


def prepare_canonical_fixture(tmp_path: Path) -> tuple[Path, Path]:
    raw_root = tmp_path / "storage/raw"
    raw_relative_path = Path("nhl_web/manual/2025-01-11/score_fixture.json")
    raw_file = raw_root / raw_relative_path
    raw_file.parent.mkdir(parents=True)
    raw_file.write_bytes(FIXTURE.read_bytes())

    raw_sha256 = hashlib.sha256(raw_file.read_bytes()).hexdigest()

    manifest_path = tmp_path / "storage/manifests/local/imports.jsonl"
    manifest_path.parent.mkdir(parents=True)

    manifest_path.write_text(
        json.dumps(
            {
                "requested_date": "2025-01-11",
                "raw_relative_path": raw_relative_path.as_posix(),
                "sha256": raw_sha256,
                "imported_at_utc": "2025-01-12T12:00:00+00:00",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    interim_root = tmp_path / "data/interim"

    canonicalize_score_file(
        raw_file=raw_file,
        raw_root=raw_root,
        import_manifest_path=manifest_path,
        output_root=interim_root,
    )

    canonical_root = interim_root / "nhl_web"
    return raw_root, canonical_root


def test_builds_stratified_five_game_pilot(
    tmp_path: Path,
) -> None:
    raw_root, canonical_root = prepare_canonical_fixture(tmp_path)

    result = build_five_game_pilot(
        canonical_root=canonical_root,
        raw_root=raw_root,
        output_root=tmp_path / "data/interim/pilot",
        audit_path=tmp_path / "audits/pilot.json",
    )

    assert result.selected_game_count == 5
    assert result.regulation_count == 3
    assert result.overtime_only_count == 1
    assert result.shootout_count == 1
    assert result.missing_required_outcome_types == ()
    assert result.status == "complete_outcome_coverage"


def test_team_ids_are_deduplicated(tmp_path: Path) -> None:
    raw_root, canonical_root = prepare_canonical_fixture(tmp_path)

    result = build_five_game_pilot(
        canonical_root=canonical_root,
        raw_root=raw_root,
        output_root=tmp_path / "data/interim/pilot",
        audit_path=tmp_path / "audits/pilot.json",
    )

    team_records = [
        json.loads(line)
        for line in Path(result.teams_path).read_text(encoding="utf-8").splitlines()
    ]

    team_ids = [record["team_id"] for record in team_records]

    assert len(team_ids) == len(set(team_ids))
    assert len(team_ids) == 8


def test_team_abbreviations_are_observations_not_identity(
    tmp_path: Path,
) -> None:
    raw_root, canonical_root = prepare_canonical_fixture(tmp_path)

    result = build_five_game_pilot(
        canonical_root=canonical_root,
        raw_root=raw_root,
        output_root=tmp_path / "data/interim/pilot",
        audit_path=tmp_path / "audits/pilot.json",
    )

    records = [
        json.loads(line)
        for line in Path(result.teams_path).read_text(encoding="utf-8").splitlines()
    ]

    team_one = next(record for record in records if record["team_id"] == "1")

    assert team_one["abbreviations"] == ["AAA"]
    assert team_one["identity_scope"] == "pilot_observation_only"
    assert len(team_one["source_game_ids"]) == 2


def test_pilot_build_is_deterministic(tmp_path: Path) -> None:
    raw_root, canonical_root = prepare_canonical_fixture(tmp_path)
    output_root = tmp_path / "data/interim/pilot"
    audit_path = tmp_path / "audits/pilot.json"

    first = build_five_game_pilot(
        canonical_root=canonical_root,
        raw_root=raw_root,
        output_root=output_root,
        audit_path=audit_path,
    )

    first_games = Path(first.selected_games_path).read_bytes()
    first_teams = Path(first.teams_path).read_bytes()

    second = build_five_game_pilot(
        canonical_root=canonical_root,
        raw_root=raw_root,
        output_root=output_root,
        audit_path=audit_path,
    )

    assert Path(second.selected_games_path).read_bytes() == first_games
    assert Path(second.teams_path).read_bytes() == first_teams
