"""Point-in-time ticker resolution."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from rebalance.data.russell import load_announcements
from rebalance.data.securities import load_security_master, resolve_security_id
from rebalance.errors import DataContractError
from tests.conftest import et


def _master(tmp_path: Path) -> pd.DataFrame:
    path = tmp_path / "master.csv"
    path.write_text(
        "\n".join(
            [
                "security_id,ticker,valid_from,valid_to,company_name",
                "SEC1,ABC,2020-01-01,2024-05-31,Abc Old",
                "SEC1,ABC2,2024-06-01,2200-01-01,Abc New",
                "SEC2,ABC,2024-06-01,2200-01-01,Other Abc",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return load_security_master(path)


def test_ticker_reuse_is_resolved_asof(tmp_path) -> None:
    master = _master(tmp_path)
    assert resolve_security_id(master, "ABC", et("2024-05-22 16:30")) == "SEC1"
    assert resolve_security_id(master, "ABC2", et("2024-06-15 16:30")) == "SEC1"
    assert resolve_security_id(master, "ABC", et("2024-06-15 16:30")) == "SEC2"
    with pytest.raises(DataContractError):
        resolve_security_id(master, "ABC", et("2019-01-01 12:00"))


def test_loader_maps_published_ticker(tmp_path) -> None:
    master = _master(tmp_path)
    path = tmp_path / "events.csv"
    path.write_text(
        "\n".join(
            [
                "published_at,effective_close,ticker,company_name,index_name,action,version,rebalance_id,source_url,retrieved_at",
                "2024-05-22T16:30:00-04:00,2024-06-28,ABC,Abc Old,RUSSELL_2000,ADD,1,2024-06,https://example.invalid/a,2024-08-01T12:00:00-04:00",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    loaded = load_announcements(path, master)
    assert loaded.iloc[0]["security_id"] == "SEC1"
    assert loaded.iloc[0]["ticker_published"] == "ABC"
    assert loaded.iloc[0]["source_hash"]
