"""Reproducible event-study runner.

Writes derived artifacts under data/processed and runs/. Refuses to write
under data/raw. The sample configuration points at synthetic fixtures and
does not estimate a live trading rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from rebalance.calendar import TradingCalendar
from rebalance.config import load_cost_config, load_research_config, repo_root, resolve_config_path
from rebalance.data.prices import load_borrow, load_fundamentals, load_prices
from rebalance.data.russell import load_announcements
from rebalance.data.securities import load_security_master
from rebalance.data.validation import assert_output_not_raw
from rebalance.errors import DataContractError
from rebalance.events.timeline import select_tradable_events
from rebalance.features.liquidity import compute_liquidity_features
from rebalance.features.pressure import compute_pressure
from rebalance.portfolio.costs import FlatBpsCostModel
from rebalance.portfolio.simulator import simulate_portfolio
from rebalance.reporting.plots import (
    plot_action_means,
    plot_by_year,
    plot_car_paths,
    plot_gross_vs_net,
    plot_quintiles,
)
from rebalance.reporting.summary import (
    assign_research_split,
    research_split_counts,
    sample_counts,
    summarize_by_rebalance,
    summarize_by_year,
    summarize_costs,
    summarize_windows,
)
from rebalance.research.cross_section import assign_pressure_quintiles, quintile_car_table
from rebalance.research.event_study import build_event_panel, car_paths, mean_abnormal_return_by_offset, window_cars
from rebalance.research.statistics import INFERENCE_NOTE

logger = logging.getLogger("rebalance")


@dataclass(frozen=True)
class RunResult:
    run_id: str
    run_dir: Path
    event_panel_path: Path
    manifest_path: Path


def run_study(config_path: Path | None = None, cost_path: Path | None = None) -> RunResult:
    root = repo_root()
    config = load_research_config(config_path)
    costs = load_cost_config(cost_path)
    if config.cost_scenario not in costs.scenarios_bps:
        raise DataContractError(f"cost scenario {config.cost_scenario} is not in costs.yaml")
    raw_root = root / "data" / "raw"
    calendar = TradingCalendar(config.calendar_name, config.timezone)
    announcements_path = resolve_config_path(root, config.paths.announcements)
    master_path = resolve_config_path(root, config.paths.security_master)
    prices_path = resolve_config_path(root, config.paths.prices)
    master = load_security_master(master_path)
    announcements = load_announcements(
        announcements_path,
        master,
        naive_timestamps_tz=config.naive_timestamps_tz,
        timezone=config.timezone,
    )
    announcements = announcements[announcements["index_name"].isin(config.index_names)].reset_index(drop=True)
    if announcements.empty:
        raise DataContractError("no announcements matched index_names")
    prices = load_prices(prices_path, calendar=calendar)
    fundamentals = None
    if config.paths.fundamentals:
        fundamentals = load_fundamentals(
            resolve_config_path(root, config.paths.fundamentals),
            naive_timestamps_tz=config.naive_timestamps_tz,
        )
    borrow = None
    if config.paths.borrow:
        borrow = load_borrow(resolve_config_path(root, config.paths.borrow), naive_timestamps_tz=config.naive_timestamps_tz)

    signals, exclusions = select_tradable_events(announcements, calendar)
    if signals.empty:
        raise DataContractError("no tradable events after point-in-time revision filtering")
    signals["research_split"] = signals["effective_close"].map(lambda day: assign_research_split(day, config.splits))
    panel, panel_diagnostics = build_event_panel(
        signals,
        prices,
        calendar,
        config.benchmark_security_id,
    )
    panel["research_split"] = panel["effective_close"].map(lambda day: assign_research_split(day, config.splits))
    cars = window_cars(panel, calendar, config.windows)
    cars["research_split"] = cars["effective_close"].map(lambda day: assign_research_split(day, config.splits))
    offset_means = mean_abnormal_return_by_offset(panel)
    liquidity, liquidity_diagnostics = compute_liquidity_features(
        signals,
        prices,
        calendar,
        config.benchmark_security_id,
        fundamentals,
    )
    pressure = compute_pressure(liquidity)
    primary = cars[cars["window"] == config.primary_window].merge(
        pressure[
            [
                "rebalance_id",
                "security_id",
                "pressure_score",
                "pressure_kind",
                "pressure_is_proxy",
                "adv20",
                "estimated_flow",
            ]
        ],
        on=["rebalance_id", "security_id"],
        how="left",
    )
    quintiles = assign_pressure_quintiles(primary)
    quintile_table = quintile_car_table(quintiles)
    window_summary = summarize_windows(
        cars,
        n_resamples=config.bootstrap_resamples,
        seed=config.bootstrap_seed,
        confidence_level=config.confidence_level,
    )
    by_rebalance = summarize_by_rebalance(cars, config.primary_window)
    by_year = summarize_by_year(by_rebalance)
    counts = sample_counts(announcements, signals, exclusions, panel_diagnostics, cars)
    split_counts = research_split_counts(signals)

    portfolio_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []
    signal_book = signals.merge(
        pressure[["rebalance_id", "security_id", "pressure_score", "pressure_kind", "pressure_is_proxy", "adv20"]],
        on=["rebalance_id", "security_id"],
        how="left",
    )
    for mode in config.portfolio_modes:
        for scheme in config.sizing_schemes:
            for scenario, bps in costs.scenarios_bps.items():
                result = simulate_portfolio(
                    signal_book,
                    prices,
                    calendar,
                    mode=mode,
                    scheme=scheme,
                    cost_model=FlatBpsCostModel(bps=bps, name=scenario),
                    max_name_weight=config.max_name_weight,
                    gross_exposure_cap=config.gross_exposure_cap,
                    benchmark_security_id=config.benchmark_security_id,
                    borrow=borrow,
                    portfolio_nav=config.portfolio_nav,
                    max_adv_participation=config.max_adv_participation,
                )
                if not result.rebalance_returns.empty:
                    portfolio_frames.append(result.rebalance_returns)
                if not result.trades.empty:
                    trade_frames.append(result.trades)
                for note in result.notes:
                    logger.info("%s %s %s: %s", mode, scheme, scenario, note)
    portfolio = pd.concat(portfolio_frames, ignore_index=True) if portfolio_frames else pd.DataFrame()
    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    cost_summary = summarize_costs(portfolio)
    gross_vs_net = _gross_vs_net_plot_frame(cost_summary, config.cost_scenario)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    run_dir = resolve_config_path(root, config.paths.run_dir) / run_id
    panel_path = resolve_config_path(root, config.paths.event_panel)
    for output in (run_dir, panel_path):
        assert_output_not_raw(output, raw_root)
    run_dir.mkdir(parents=True, exist_ok=False)
    panel_path.parent.mkdir(parents=True, exist_ok=True)

    panel.to_parquet(panel_path, index=False)
    panel.to_parquet(run_dir / "event_panel.parquet", index=False)
    cars.to_parquet(run_dir / "window_cars.parquet", index=False)
    pressure.to_parquet(run_dir / "liquidity_features.parquet", index=False)
    quintiles.to_parquet(run_dir / "pressure_quintiles.parquet", index=False)
    _write_frame(window_summary, run_dir / "window_summary.csv")
    _write_frame(by_rebalance, run_dir / "by_rebalance.csv")
    _write_frame(by_year, run_dir / "by_year.csv")
    _write_frame(quintile_table, run_dir / "quintile_summary.csv")
    _write_frame(counts, run_dir / "sample_counts.csv")
    _write_frame(split_counts, run_dir / "research_splits.csv")
    _write_frame(cost_summary, run_dir / "cost_summary.csv")
    _write_frame(offset_means, run_dir / "mean_ar_by_offset.csv")
    _write_frame(exclusions, run_dir / "exclusions.csv")
    _write_frame(panel_diagnostics, run_dir / "panel_diagnostics.csv")
    _write_frame(liquidity_diagnostics, run_dir / "liquidity_diagnostics.csv")
    if not trades.empty:
        trades.to_parquet(run_dir / "trades.parquet", index=False)
    if not portfolio.empty:
        portfolio.to_parquet(run_dir / "rebalance_returns.parquet", index=False)

    figure_dir = run_dir / "figures"
    announcement_paths = car_paths(panel, align="days_from_announcement", start_offset=0, end_offset=15, stop_at_rebalance=True)
    rebalance_paths = car_paths(panel, align="days_from_rebalance", start_offset=-5, end_offset=20)
    plot_car_paths(
        announcement_paths,
        figure_dir / "car_from_announcement.png",
        xlabel="Trading sessions from first tradable open",
        title="CAR from entry, stopped at the rebalance close",
    )
    plot_car_paths(
        rebalance_paths,
        figure_dir / "car_around_rebalance.png",
        xlabel="Trading sessions from rebalance close",
        title="CAR from five sessions before rebalance through twenty after",
    )
    plot_action_means(window_summary, figure_dir / "additions_vs_deletions.png", window=config.primary_window)
    plot_quintiles(quintile_table, figure_dir / "pressure_quintiles.png")
    plot_by_year(by_year, figure_dir / "car_by_year.png")
    plot_gross_vs_net(gross_vs_net, figure_dir / "gross_vs_net.png")

    manifest = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(root),
        "configuration": config.model_dump(mode="json"),
        "cost_configuration": costs.model_dump(mode="json"),
        "event_dataset_version": config.event_dataset_version,
        "price_dataset_version": config.price_dataset_version,
        "event_file_sha256": _sha256(announcements_path),
        "price_file_sha256": _sha256(prices_path),
        "security_master_sha256": _sha256(master_path),
        "date_range": {
            "first_published_at": _iso(signals["signal_known_at"].min()),
            "last_published_at": _iso(signals["signal_known_at"].max()),
            "first_effective_close": str(pd.Timestamp(signals["effective_close"].min()).date()),
            "last_effective_close": str(pd.Timestamp(signals["effective_close"].max()).date()),
        },
        "entry_rule": config.entry_rule,
        "exit_rule": config.exit_rule,
        "cost_assumptions": costs.scenarios_bps,
        "primary_cost_scenario": config.cost_scenario,
        "inference_note": INFERENCE_NOTE,
        "profitability_claim": False,
        "sample_note": (
            "Outputs describe the input files. Synthetic fixtures are not evidence "
            "about Russell reconstitutions."
        ),
    }
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    logger.info("wrote %s", manifest_path)
    return RunResult(run_id=run_id, run_dir=run_dir, event_panel_path=panel_path, manifest_path=manifest_path)


def _gross_vs_net_plot_frame(cost_summary: pd.DataFrame, highlighted: str) -> pd.DataFrame:
    if cost_summary.empty:
        return cost_summary
    subset = cost_summary[(cost_summary["scheme"] == "equal") & (cost_summary["cost_model"] == highlighted)]
    return subset.reset_index(drop=True)


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    commit = completed.stdout.strip()
    return commit or None


def _iso(value: object) -> str:
    return pd.Timestamp(value).isoformat()


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Run the reconstitution event study")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--costs", type=Path, default=None)
    args = parser.parse_args(argv)
    result = run_study(args.config, args.costs)
    print(result.run_dir)


if __name__ == "__main__":
    main()
