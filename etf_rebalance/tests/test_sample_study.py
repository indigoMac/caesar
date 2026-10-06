"""End-to-end run on the synthetic fixtures. Numbers are not empirical results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from rebalance.config import repo_root
from rebalance.research.run import run_study


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_sample_study_is_point_in_time_and_reproducible() -> None:
    root = repo_root()
    raw_files = [
        root / "data/raw/russell/sample_announcements.csv",
        root / "data/raw/russell/sample_security_master.csv",
        root / "data/raw/russell/sample_fundamentals.csv",
        root / "data/raw/prices/sample_prices.csv",
    ]
    before = {path: _sha256(path) for path in raw_files}
    result = run_study(root / "config/research.yaml", root / "config/costs.yaml")
    after = {path: _sha256(path) for path in raw_files}
    assert before == after

    panel = pd.read_parquet(result.event_panel_path)
    required = {
        "rebalance_id",
        "security_id",
        "action",
        "date",
        "days_from_announcement",
        "days_from_rebalance",
        "stock_return",
        "benchmark_return",
        "abnormal_return",
        "cumulative_abnormal_return",
        "signal_known_at",
        "feature_asof",
    }
    assert required <= set(panel.columns)
    assert "X01" not in set(panel["security_id"])
    assert set(panel["action"]) == {"ADD", "DELETE"}
    june_2024 = panel[panel["rebalance_id"] == "2024-06"]
    assert set(pd.to_datetime(june_2024["entry_session"]).dt.strftime("%Y-%m-%d")) == {"2024-05-28"}

    exclusions = pd.read_csv(result.run_dir / "exclusions.csv")
    cancelled = exclusions[exclusions["security_id"] == "X01"]
    assert cancelled.iloc[0]["reason"] == "cancelled_before_entry"

    features = pd.read_parquet(result.run_dir / "liquidity_features.parquet")
    a01 = features[(features["security_id"] == "A01") & (features["rebalance_id"] == "2024-06")].iloc[0]
    assert a01["market_cap"] == 1_000_000_000.0
    assert a01["pressure_is_proxy"]
    assert a01["pressure_kind"] == "abs_delta_weight_over_adv20"
    assert pd.isna(a01["estimated_flow"])

    summary = pd.read_csv(result.run_dir / "window_summary.csv")
    primary = summary[(summary["action"] == "ADD") & (summary["window"] == "ann_to_rebalance")].iloc[0]
    assert primary["n_rebalances"] == 2
    assert primary["n_names"] == 10
    assert primary["event_t_df"] == 1
    assert "not stock-level" in primary["inference"]

    quintiles = pd.read_csv(result.run_dir / "quintile_summary.csv")
    assert {"Q1", "Q5"} <= set(quintiles["pressure_quintile"])

    costs = pd.read_csv(result.run_dir / "cost_summary.csv")
    shorts = costs[costs["mode"] == "short_deletions"]
    assert shorts["theoretical_short_books"].all()
    assert (result.run_dir / "figures" / "car_around_rebalance.png").stat().st_size > 0

    manifest = json.loads((result.run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    for key in (
        "run_id",
        "timestamp",
        "git_commit",
        "configuration",
        "event_dataset_version",
        "price_dataset_version",
        "date_range",
        "entry_rule",
        "exit_rule",
        "cost_assumptions",
    ):
        assert key in manifest
    assert manifest["profitability_claim"] is False
    assert manifest["event_file_sha256"] == before[raw_files[0]]
