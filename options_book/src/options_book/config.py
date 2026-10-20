"""Load the paper book from YAML. Paths are relative to the package root."""

from __future__ import annotations

from datetime import time
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class BrokerSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    host: str = "127.0.0.1"
    port: int = 4002
    client_id: int = 17
    live_enabled: bool = False
    submit_orders: bool = False
    account: str = ""

    @field_validator("port")
    @classmethod
    def port_range(cls, value: int) -> int:
        if not 1 <= value <= 65535:
            raise ValueError("port out of range")
        return value

    @field_validator("client_id")
    @classmethod
    def client_range(cls, value: int) -> int:
        if not 1 <= value <= 999:
            raise ValueError("client_id must be from 1 to 999")
        return value


class SessionSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    timezone: Literal["America/New_York"] = "America/New_York"
    entry_start: str = "09:45"
    entry_end: str = "15:45"
    calendar: Literal["XNYS"] = "XNYS"

    @model_validator(mode="after")
    def window_is_open(self) -> Self:
        start = time.fromisoformat(self.entry_start)
        end = time.fromisoformat(self.entry_end)
        if start >= end:
            raise ValueError("entry window is empty")
        return self


class SpreadSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    quantity: Literal[1] = 1
    target_dte: int = 40
    min_dte: int = 30
    max_dte: int = 45
    short_delta: float = -0.16
    width_points: float = 25
    min_credit_points: float = 0.50
    max_loss_dollars: float = 2500
    profit_take_fraction: float = 0.50
    close_dte: int = 21
    loss_multiple_of_credit: float = 2.0
    order_timeout_seconds: int = 180
    scan_points_below_spot: float = 400
    option_tick: float = 0.05

    @model_validator(mode="after")
    def check_rules(self) -> Self:
        if not 0 < self.min_dte <= self.target_dte <= self.max_dte:
            raise ValueError("target_dte must sit inside a positive dte window")
        if self.close_dte < 0 or self.close_dte >= self.min_dte:
            raise ValueError("close_dte must be below min_dte")
        if not -1 < self.short_delta < 0:
            raise ValueError("short_delta must be a negative put delta")
        if self.width_points <= 0:
            raise ValueError("width_points must be positive")
        if self.min_credit_points <= 0:
            raise ValueError("min_credit_points must be positive")
        if self.max_loss_dollars <= 0:
            raise ValueError("max_loss_dollars must be positive")
        if not 0 < self.profit_take_fraction < 1:
            raise ValueError("profit_take_fraction must be between 0 and 1")
        if self.loss_multiple_of_credit <= 0:
            raise ValueError("loss_multiple_of_credit must be positive")
        if self.order_timeout_seconds <= 0:
            raise ValueError("order_timeout_seconds must be positive")
        if self.scan_points_below_spot <= self.width_points:
            raise ValueError("scan window must be wider than the spread")
        if self.option_tick <= 0:
            raise ValueError("option_tick must be positive")
        return self


class HedgeSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    symbol: Literal["MES", "ES"] = "MES"
    band_contracts: float = 1
    min_adjustment: int = 1
    max_mes_position: int = 10
    greek_max_age_seconds: float = 5
    tick_size: float = 0.25
    max_spread_points: float = 1

    @model_validator(mode="after")
    def check_rules(self) -> Self:
        if self.band_contracts < 1:
            raise ValueError("band_contracts must be at least 1")
        if self.min_adjustment < 1:
            raise ValueError("min_adjustment must be at least 1")
        if self.max_mes_position < 1:
            raise ValueError("max_mes_position must be at least 1")
        if self.greek_max_age_seconds <= 0:
            raise ValueError("greek_max_age_seconds must be positive")
        if self.tick_size <= 0:
            raise ValueError("tick_size must be positive")
        if self.max_spread_points <= 0:
            raise ValueError("max_spread_points must be positive")
        return self


class RiskSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_daily_loss_dollars: float = 2500
    halt_file: str = "data/HALT"

    @field_validator("max_daily_loss_dollars")
    @classmethod
    def loss_limit(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("max_daily_loss_dollars must be positive")
        return value


class LoopSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    poll_seconds: int = 60

    @field_validator("poll_seconds")
    @classmethod
    def poll_is_slow(cls, value: int) -> int:
        if value < 1:
            raise ValueError("poll_seconds must be at least 1")
        return value


class StoreSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = "data/state.sqlite"


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    broker: BrokerSettings = Field(default_factory=BrokerSettings)
    session: SessionSettings = Field(default_factory=SessionSettings)
    spread: SpreadSettings = Field(default_factory=SpreadSettings)
    hedge: HedgeSettings = Field(default_factory=HedgeSettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    loop: LoopSettings = Field(default_factory=LoopSettings)
    store: StoreSettings = Field(default_factory=StoreSettings)


class LoadedConfig:
    """Settings plus the config path used to resolve data files."""

    def __init__(self, settings: Settings, config_path: Path) -> None:
        self.settings = settings
        self.config_path = config_path

    def resolve(self, relative: str) -> Path:
        path = Path(relative)
        if path.is_absolute():
            return path
        return self.config_path.resolve().parent.parent / path


def load_settings(path: Path) -> LoadedConfig:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path} does not contain a mapping")
    return LoadedConfig(Settings.model_validate(raw), path)
