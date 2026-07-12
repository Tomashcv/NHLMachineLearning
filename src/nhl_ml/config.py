"""Typed project configuration and source-registry loading."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a project configuration file is invalid."""


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping using PyYAML's safe loader."""
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")

    with path.open("r", encoding="utf-8") as handle:
        payload: Any = yaml.safe_load(handle)

    if not isinstance(payload, dict):
        raise ConfigError(f"Expected a YAML mapping in {path}")

    return payload


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    """Validated high-level project configuration."""

    project_name: str
    canonical_timezone: str
    primary_market: str
    early_cutoff_minutes: int
    pregame_cutoff_minutes: int
    pilot_target_games: int
    pilot_minimum_seasons: int

    @classmethod
    def from_path(cls, path: Path) -> "ProjectConfig":
        """Build and validate project configuration from YAML."""
        payload = load_yaml(path)

        try:
            project = payload["project"]
            cutoffs = payload["decision_cutoffs"]
            pilot = payload["pilot"]

            config = cls(
                project_name=str(project["name"]),
                canonical_timezone=str(project["canonical_timezone"]),
                primary_market=str(project["primary_market"]),
                early_cutoff_minutes=int(cutoffs["early_minutes_before_start"]),
                pregame_cutoff_minutes=int(cutoffs["pregame_minutes_before_start"]),
                pilot_target_games=int(pilot["target_games"]),
                pilot_minimum_seasons=int(pilot["minimum_seasons"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigError(f"Invalid project configuration in {path}: {exc}") from exc

        config.validate()
        return config

    def validate(self) -> None:
        """Validate project-level invariants."""
        if not self.project_name.strip():
            raise ConfigError("Project name cannot be empty")

        if self.canonical_timezone != "UTC":
            raise ConfigError("The canonical project timezone must be UTC")

        if not self.primary_market.strip():
            raise ConfigError("Primary market cannot be empty")

        if self.pregame_cutoff_minutes <= 0:
            raise ConfigError("Pregame cutoff must be positive")

        if self.early_cutoff_minutes <= self.pregame_cutoff_minutes:
            raise ConfigError("Early cutoff must occur before the pregame cutoff")

        if not 20 <= self.pilot_target_games <= 50:
            raise ConfigError("Pilot target must contain between 20 and 50 games")

        if self.pilot_minimum_seasons < 2:
            raise ConfigError("Pilot must cover at least two seasons")


@dataclass(frozen=True, slots=True)
class SourceEntry:
    """One audited or candidate data source."""

    source_id: str
    name: str
    category: str
    audit_status: str
    automation_status: str
    redistribution_status: str
    research_only: bool
    notes: str

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "SourceEntry":
        """Create a source entry from a YAML mapping."""
        required_fields = {
            "source_id",
            "name",
            "category",
            "audit_status",
            "automation_status",
            "redistribution_status",
            "research_only",
            "notes",
        }

        missing = sorted(required_fields - raw.keys())
        if missing:
            raise ConfigError(f"Source entry is missing fields: {missing}")

        research_only = raw["research_only"]
        if not isinstance(research_only, bool):
            raise ConfigError("research_only must be a YAML boolean")

        entry = cls(
            source_id=str(raw["source_id"]).strip(),
            name=str(raw["name"]).strip(),
            category=str(raw["category"]).strip(),
            audit_status=str(raw["audit_status"]).strip(),
            automation_status=str(raw["automation_status"]).strip(),
            redistribution_status=str(raw["redistribution_status"]).strip(),
            research_only=research_only,
            notes=str(raw["notes"]).strip(),
        )

        entry.validate()
        return entry

    def validate(self) -> None:
        """Validate mandatory source metadata."""
        string_fields = {
            "source_id": self.source_id,
            "name": self.name,
            "category": self.category,
            "audit_status": self.audit_status,
            "automation_status": self.automation_status,
            "redistribution_status": self.redistribution_status,
            "notes": self.notes,
        }

        empty = sorted(name for name, value in string_fields.items() if not value)
        if empty:
            raise ConfigError(f"Source fields cannot be empty: {empty}")


@dataclass(frozen=True, slots=True)
class SourceRegistry:
    """Collection of uniquely identified source entries."""

    entries: tuple[SourceEntry, ...]

    @classmethod
    def from_path(cls, path: Path) -> "SourceRegistry":
        """Load a source registry from YAML."""
        payload = load_yaml(path)
        raw_sources = payload.get("sources")

        if not isinstance(raw_sources, list) or not raw_sources:
            raise ConfigError("Source registry must contain a non-empty sources list")

        entries = tuple(SourceEntry.from_mapping(raw) for raw in raw_sources)
        source_ids = [entry.source_id for entry in entries]

        duplicates = sorted(
            source_id for source_id in set(source_ids) if source_ids.count(source_id) > 1
        )

        if duplicates:
            raise ConfigError(f"Duplicate source IDs: {duplicates}")

        return cls(entries=entries)

    def by_id(self, source_id: str) -> SourceEntry:
        """Return one source by its stable identifier."""
        for entry in self.entries:
            if entry.source_id == source_id:
                return entry

        raise KeyError(f"Unknown source ID: {source_id}")
