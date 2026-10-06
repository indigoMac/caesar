"""Tables for window CARs, sample counts, and cost-adjusted portfolio returns."""

from __future__ import annotations

from datetime import date

import pandas as pd

from rebalance.config import DateRange
from rebalance.research.statistics import (
    INFERENCE_NOTE,
    bootstrap_event_means,
    event_level_means,
    event_level_t_interval,
    pooled_clustered_mean,
)


def assign_research_split(effective_close: pd.Timestamp, splits: dict[str, DateRange]) -> str:
    day = pd.Timestamp(effective_close).date()
    for name, bounds in splits.items():
        if bounds.start <= day <= bounds.end:
            return name
    return "unassigned"


def summarize_windows(
    window_cars: pd.DataFrame,
    *,
    n_resamples: int,
    seed: int,
    confidence_level: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if window_cars.empty:
        return pd.DataFrame()
    for (action, window), group in window_cars.groupby(["action", "window"], sort=False):
        usable = group.dropna(subset=["car"])
        if usable.empty:
            rows.append(
                {
                    "action": action,
                    "window": window,
                    "n_names": 0,
                    "n_rebalances": 0,
                    "mean_car_event_equal": float("nan"),
                    "median_car_pooled": float("nan"),
                    "inference": INFERENCE_NOTE,
                    "window_note": "no complete windows",
                }
            )
            continue
        means = event_level_means(usable, "car")
        bootstrap = bootstrap_event_means(
            usable,
            "car",
            n_resamples=n_resamples,
            seed=seed,
            confidence_level=confidence_level,
        )
        t_interval = event_level_t_interval(means, confidence_level)
        clustered = pooled_clustered_mean(usable, "car", confidence_level=confidence_level)
        excludes_zero = (
            bootstrap.n_resamples > 0
            and bootstrap.ci_low == bootstrap.ci_low
            and (bootstrap.ci_high < 0 or bootstrap.ci_low > 0)
        )
        rows.append(
            {
                "action": action,
                "window": window,
                "n_names": int(len(usable)),
                "n_rebalances": int(usable["rebalance_id"].nunique()),
                "mean_car_event_equal": bootstrap.estimate,
                "median_car_pooled": float(usable["car"].median()),
                "mean_car_pooled": float(usable["car"].mean()),
                "bootstrap_ci_low": bootstrap.ci_low,
                "bootstrap_ci_high": bootstrap.ci_high,
                "bootstrap_resamples": bootstrap.n_resamples,
                "event_t_df": t_interval.degrees_of_freedom,
                "event_t_ci_low": t_interval.ci_low,
                "event_t_ci_high": t_interval.ci_high,
                "pooled_clustered_mean": clustered.estimate,
                "pooled_clustered_se": clustered.standard_error,
                "pooled_clustered_ci_low": clustered.ci_low,
                "pooled_clustered_ci_high": clustered.ci_high,
                "pooled_clustered_n_obs": clustered.n_observations,
                "ci_excludes_zero": bool(excludes_zero),
                "inference": INFERENCE_NOTE,
                "window_note": "",
            }
        )
    return pd.DataFrame(rows)


def summarize_by_rebalance(window_cars: pd.DataFrame, primary_window: str) -> pd.DataFrame:
    subset = window_cars[window_cars["window"] == primary_window]
    if subset.empty:
        return pd.DataFrame()
    table = (
        subset.groupby(["rebalance_id", "action", "effective_close"], sort=False)
        .agg(n_names=("car", "count"), mean_car=("car", "mean"), median_car=("car", "median"))
        .reset_index()
    )
    table["year"] = pd.to_datetime(table["effective_close"]).dt.year
    return table


def summarize_by_year(by_rebalance: pd.DataFrame) -> pd.DataFrame:
    if by_rebalance.empty:
        return pd.DataFrame()
    return (
        by_rebalance.groupby(["year", "action"], sort=False)
        .agg(
            n_rebalances=("rebalance_id", "nunique"),
            n_names=("n_names", "sum"),
            mean_car_event_equal=("mean_car", "mean"),
        )
        .reset_index()
        .sort_values(["action", "year"])
    )


def sample_counts(
    announcements: pd.DataFrame,
    signals: pd.DataFrame,
    exclusions: pd.DataFrame,
    panel_diagnostics: pd.DataFrame,
    window_cars: pd.DataFrame,
) -> pd.DataFrame:
    exclusion_counts = (
        exclusions["reason"].value_counts().rename_axis("item").reset_index(name="n")
        if not exclusions.empty and "reason" in exclusions.columns
        else pd.DataFrame(columns=["item", "n"])
    )
    rows = [
        {"item": "announcement_rows", "n": int(len(announcements))},
        {"item": "tradable_events", "n": int(len(signals))},
        {"item": "tradable_additions", "n": int((signals["action"] == "ADD").sum()) if not signals.empty else 0},
        {"item": "tradable_deletions", "n": int((signals["action"] == "DELETE").sum()) if not signals.empty else 0},
        {"item": "rebalances", "n": int(signals["rebalance_id"].nunique()) if not signals.empty else 0},
        {"item": "exclusions", "n": int(len(exclusions))},
        {"item": "panel_diagnostics", "n": int(len(panel_diagnostics))},
        {
            "item": "incomplete_primary_windows",
            "n": int((window_cars["window_status"] != "ok").sum()) if not window_cars.empty else 0,
        },
    ]
    base = pd.DataFrame(rows)
    if not exclusion_counts.empty:
        exclusion_counts["item"] = "exclusion:" + exclusion_counts["item"].astype(str)
        base = pd.concat([base, exclusion_counts], ignore_index=True)
    return base


def summarize_costs(rebalance_returns: pd.DataFrame) -> pd.DataFrame:
    if rebalance_returns.empty:
        return pd.DataFrame()
    table = (
        rebalance_returns.groupby(["mode", "scheme", "cost_model"], sort=False)
        .agg(
            n_rebalances=("rebalance_id", "nunique"),
            mean_gross_return=("gross_return", "mean"),
            mean_net_return=("net_return", "mean"),
            theoretical_short_books=("contains_theoretical_shorts", "max"),
        )
        .reset_index()
    )
    table["cost_note"] = table["theoretical_short_books"].map(
        lambda flag: "short leg is theoretical without located borrow" if bool(flag) else "no short leg"
    )
    return table


def research_split_counts(signals: pd.DataFrame) -> pd.DataFrame:
    if signals.empty or "research_split" not in signals.columns:
        return pd.DataFrame(columns=["research_split", "n"])
    return signals["research_split"].value_counts().rename_axis("research_split").reset_index(name="n")


def year_of(day: date | pd.Timestamp) -> int:
    return pd.Timestamp(day).year
