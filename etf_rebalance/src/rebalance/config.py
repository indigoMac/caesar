"""Load and validate research and cost configuration."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from rebalance.errors import DataContractError

Anchor = Literal["entry", "rebalance"]
PortfolioMode = Literal[
    "long_additions",
    "short_deletions",
    "long_short",
    "long_additions_hedged",
]
SizingScheme = Literal["equal", "pressure"]


class DateRange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: date
    end: date

    @model_validator(mode="after")
    def _ordered(self) -> DateRange:
        if self.end < self.start:
            raise ValueError(f"split end {self.end} is before start {self.start}")
        return self


class WindowSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    start_anchor: Anchor
    start_offset: int
    end_anchor: Anchor
    end_offset: int


class PathsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    announcements: str
    security_master: str
    prices: str
    fundamentals: str | None = None
    borrow: str | None = None
    event_panel: str = "data/processed/event_panel.parquet"
    run_dir: str = "runs"


class ResearchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    calendar_name: str = "XNYS"
    timezone: str = "America/New_York"
    benchmark_security_id: str
    index_names: list[str] = Field(min_length=1)
    primary_window: str
    windows: list[WindowSpec] = Field(min_length=1)
    bootstrap_resamples: int = Field(ge=1)
    bootstrap_seed: int
    confidence_level: float = Field(gt=0, lt=1)
    max_name_weight: float = Field(gt=0, le=1)
    gross_exposure_cap: float = Field(gt=0)
    portfolio_nav: float | None = None
    max_adv_participation: float | None = None
    entry_rule: str
    exit_rule: str
    event_dataset_version: str
    price_dataset_version: str
    naive_timestamps_tz: str | None = None
    portfolio_modes: list[PortfolioMode] = Field(min_length=1)
    sizing_schemes: list[SizingScheme] = Field(min_length=1)
    cost_scenario: str
    splits: dict[str, DateRange]
    paths: PathsConfig

    @model_validator(mode="after")
    def _window_and_liquidity(self) -> ResearchConfig:
        names = [window.name for window in self.windows]
        if len(names) != len(set(names)):
            raise ValueError("window names must be unique")
        if self.primary_window not in names:
            raise ValueError(f"primary_window {self.primary_window} is not defined")
        nav_set = self.portfolio_nav is not None
        adv_set = self.max_adv_participation is not None
        if nav_set != adv_set:
            raise ValueError(
                "portfolio_nav and max_adv_participation must both be set or both be null"
            )
        if self.portfolio_nav is not None and self.portfolio_nav <= 0:
            raise ValueError("portfolio_nav must be positive")
        if self.max_adv_participation is not None and self.max_adv_participation <= 0:
            raise ValueError("max_adv_participation must be positive")
        return self


class CostConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenarios_bps: dict[str, float]

    @model_validator(mode="after")
    def _non_negative(self) -> CostConfig:
        if not self.scenarios_bps:
            raise ValueError("at least one cost scenario is required")
        for name, bps in self.scenarios_bps.items():
            if bps < 0:
                raise ValueError(f"cost scenario {name} has negative bps")
        return self


def repo_root() -> Path:
    """Locate the etf_rebalance directory that contains this package."""
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").exists() and (candidate / "src" / "rebalance").exists():
            return candidate
    raise DataContractError("could not locate the etf_rebalance repository root")


def _load_yaml(path: Path) -> dict:
    if not path.is_file():
        raise DataContractError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise DataContractError(f"config file must be a mapping: {path}")
    return loaded


def load_research_config(path: Path | None = None) -> ResearchConfig:
    config_path = path or (repo_root() / "config" / "research.yaml")
    return ResearchConfig.model_validate(_load_yaml(config_path))


def load_cost_config(path: Path | None = None) -> CostConfig:
    config_path = path or (repo_root() / "config" / "costs.yaml")
    return CostConfig.model_validate(_load_yaml(config_path))


def resolve_config_path(root: Path, relative_or_absolute: str) -> Path:
    path = Path(relative_or_absolute)
    if path.is_absolute():
        return path
    return (root / path).resolve()
