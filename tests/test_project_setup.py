from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_required_directories_exist() -> None:
    required_directories = [
        "src/nhl_ml",
        "tests",
        "configs",
        "scripts",
        "docs",
        "data/raw",
        "storage/manifests",
        "storage/audits",
    ]

    missing = [
        relative_path
        for relative_path in required_directories
        if not (PROJECT_ROOT / relative_path).is_dir()
    ]

    assert not missing, f"Missing required directories: {missing}"


def test_private_environment_file_is_not_committed() -> None:
    assert not (PROJECT_ROOT / ".env").exists()
