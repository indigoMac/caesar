"""Write the four research notebooks. They load the synthetic fixtures only."""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"

SETUP = """\
from pathlib import Path

import pandas as pd
from IPython.display import display

from rebalance.calendar import TradingCalendar
from rebalance.config import load_research_config, repo_root, resolve_config_path
from rebalance.data.prices import load_fundamentals, load_prices
from rebalance.data.russell import load_announcements
from rebalance.data.securities import load_security_master
from rebalance.events.revisions import event_state_asof
from rebalance.events.timeline import select_tradable_events

root = repo_root()
config = load_research_config(root / "config" / "research.yaml")
calendar = TradingCalendar(config.calendar_name, config.timezone)
master = load_security_master(resolve_config_path(root, config.paths.security_master))
events = load_announcements(
    resolve_config_path(root, config.paths.announcements),
    master,
    timezone=config.timezone,
)
trades, excluded = select_tradable_events(events, calendar)
print("Synthetic fixtures only. Do not read these prints as a Russell result.")
print(f"announcements={len(events)} tradable={len(trades)} excluded={len(excluded)}")
"""


def _write(name: str, cells: list) -> None:
    notebook = new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
    path = NOTEBOOKS / name
    nbformat.write(notebook, path)
    print(path)


def main() -> None:
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    _write(
        "01_event_data_audit.ipynb",
        [
            new_markdown_cell(
                "Audit the synthetic announcement log. "
                "`event_state_asof` hides rows that were not yet public and drops a name whose latest public action is `NONE`. "
                "The raw log still contains the earlier ADD."
            ),
            new_code_cell(SETUP),
            new_code_cell(
                """\
x01 = events.loc[events["security_id"] == "X01", ["published_at", "action", "version"]]
print("X01 revision log")
display(x01)

may26 = event_state_asof(events, pd.Timestamp("2024-05-26 12:00", tz="America/New_York"))
print("X01 actions public on 26 May:", may26.loc[may26["security_id"] == "X01", "action"].tolist())

decision = calendar.session_open("2024-05-28") - pd.Timedelta(nanoseconds=1)
at_entry = event_state_asof(events, decision)
print("X01 actions just before the 28 May open:", at_entry.loc[at_entry["security_id"] == "X01", "action"].tolist())
print("ADD rows still stored in the log:", int((events["action"] == "ADD").sum()))
print(excluded.loc[excluded["security_id"] == "X01", ["security_id", "reason"]].to_string(index=False))
print("2024 entry sessions:", sorted(trades.loc[trades["rebalance_id"] == "2024-06", "entry_session"].astype(str).unique()))
"""
            ),
        ],
    )
    _write(
        "02_event_study.ipynb",
        [
            new_markdown_cell(
                "Build the event panel and window CARs. "
                "Abnormal return is the stock total return minus the benchmark, on the same basis. "
                "Deletions are not sign-flipped here. "
                "`n_rebalances` is the sample size used for intervals."
            ),
            new_code_cell(SETUP),
            new_code_cell(
                """\
from rebalance.reporting.summary import summarize_windows
from rebalance.research.event_study import build_event_panel, window_cars

prices = load_prices(resolve_config_path(root, config.paths.prices), calendar=calendar)
panel, diagnostics = build_event_panel(trades, prices, calendar, config.benchmark_security_id)
cars = window_cars(panel, calendar, config.windows)
summary = summarize_windows(
    cars,
    n_resamples=config.bootstrap_resamples,
    seed=config.bootstrap_seed,
    confidence_level=config.confidence_level,
)
columns = ["action", "window", "n_names", "n_rebalances", "mean_car_event_equal", "event_t_df", "bootstrap_ci_low", "bootstrap_ci_high"]
print(f"panel rows={len(panel)} diagnostics={len(diagnostics)}")
print("CAR is a sum of simple abnormal returns. It is not the portfolio's compounded return.")
display(summary[columns])
"""
            ),
        ],
    )
    _write(
        "03_pressure_signal.ipynb",
        [
            new_markdown_cell(
                "Proxy pressure is `abs(delta_weight) / ADV20` until `estimated_flow` exists. "
                "The proxy is not written into `estimated_flow`. "
                "Quintiles are formed inside each reconstitution and action. Q1 is the lowest score."
            ),
            new_code_cell(SETUP),
            new_code_cell(
                """\
from rebalance.features.liquidity import compute_liquidity_features
from rebalance.features.pressure import compute_pressure
from rebalance.research.cross_section import assign_pressure_quintiles, quintile_car_table
from rebalance.research.event_study import build_event_panel, window_cars

prices = load_prices(resolve_config_path(root, config.paths.prices), calendar=calendar)
fundamentals = load_fundamentals(resolve_config_path(root, config.paths.fundamentals))
features, feature_diagnostics = compute_liquidity_features(
    trades, prices, calendar, config.benchmark_security_id, fundamentals
)
pressure = compute_pressure(features)
print(pressure["pressure_kind"].value_counts().to_string())
print("non-null estimated_flow rows:", int(pressure["estimated_flow"].notna().sum()))
print("proxy rows:", int(pressure["pressure_is_proxy"].sum()), "feature diagnostics:", len(feature_diagnostics))

panel, _ = build_event_panel(trades, prices, calendar, config.benchmark_security_id)
cars = window_cars(panel, calendar, config.windows)
primary = cars[cars["window"] == config.primary_window].merge(
    pressure[["rebalance_id", "security_id", "pressure_score", "pressure_kind", "pressure_is_proxy"]],
    on=["rebalance_id", "security_id"],
    how="left",
)
table = quintile_car_table(assign_pressure_quintiles(primary))
print("Synthetic quintile CAR. Not a test of index-demand monotonicity.")
display(table)
"""
            ),
        ],
    )
    _write(
        "04_portfolio_backtest.ipynb",
        [
            new_markdown_cell(
                "Buy-and-hold portfolio returns subtract the entry cost and the exit cost. "
                "Short deletions are theoretical when no borrow file is supplied. "
                "Gross and net figures below are fixture accounting, not a performance claim."
            ),
            new_code_cell(
                """\
import pandas as pd
from IPython.display import display

from rebalance.research.run import run_study

result = run_study()
print(result.run_dir)
costs = pd.read_csv(result.run_dir / "cost_summary.csv")
equal = costs[costs["scheme"] == "equal"]
display(equal[["mode", "cost_model", "n_rebalances", "mean_gross_return", "mean_net_return", "cost_note"]])
print("Manifest says profitability_claim is false. Leave it that way.")
"""
            ),
            new_code_cell(
                """\
from IPython.display import Image, display

display(Image(filename=str(result.run_dir / "figures" / "gross_vs_net.png")))
display(Image(filename=str(result.run_dir / "figures" / "additions_vs_deletions.png")))
"""
            ),
        ],
    )


if __name__ == "__main__":
    main()
