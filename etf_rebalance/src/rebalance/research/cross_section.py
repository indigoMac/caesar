"""Within-rebalance pressure quintiles.

Quintiles are assigned independently inside each (rebalance, action) group.
Q1 is the lowest pressure score and Q5 is the highest. Additions and
deletions are not pooled, because their abnormal returns have opposite
economic signs. Groups with fewer than five scored names are left unranked.
"""

from __future__ import annotations

import pandas as pd

from rebalance.errors import DataContractError


def assign_pressure_quintiles(
    frame: pd.DataFrame,
    *,
    score_col: str = "pressure_score",
    group_cols: tuple[str, ...] = ("rebalance_id", "action"),
) -> pd.DataFrame:
    if score_col not in frame.columns:
        raise DataContractError(f"missing pressure column {score_col}")
    for column in group_cols:
        if column not in frame.columns:
            raise DataContractError(f"missing grouping column {column}")
    parts: list[pd.DataFrame] = []
    for _, group in frame.groupby(list(group_cols), sort=False):
        scored = group.copy()
        valid = scored[score_col].notna()
        labels = pd.Series(pd.NA, index=scored.index, dtype="object")
        if int(valid.sum()) < 5:
            scored["pressure_quintile"] = pd.NA
            scored["quintile_status"] = "insufficient_names"
        else:
            ranks = scored.loc[valid, score_col].rank(method="first")
            labels.loc[valid] = pd.qcut(ranks, 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"]).astype(str)
            scored["pressure_quintile"] = labels
            scored["quintile_status"] = "ok"
        parts.append(scored)
    if not parts:
        empty = frame.copy()
        empty["pressure_quintile"] = pd.NA
        empty["quintile_status"] = pd.NA
        return empty
    return pd.concat(parts, ignore_index=True)


def quintile_car_table(quintiles: pd.DataFrame, car_col: str = "car") -> pd.DataFrame:
    """Mean and median CAR by quintile, equally weighting rebalances."""
    usable = quintiles[quintiles["quintile_status"] == "ok"].dropna(subset=[car_col, "pressure_quintile"])
    if usable.empty:
        return pd.DataFrame(
            columns=[
                "action",
                "pressure_quintile",
                "n_names",
                "n_rebalances",
                "mean_car_event_equal",
                "median_car_pooled",
                "mean_car_pooled",
            ]
        )
    by_event = (
        usable.groupby(["action", "rebalance_id", "pressure_quintile"], sort=False)[car_col]
        .agg(event_mean="mean", event_median="median", n_names="size")
        .reset_index()
    )
    summary = (
        by_event.groupby(["action", "pressure_quintile"], sort=False)
        .agg(
            n_names=("n_names", "sum"),
            n_rebalances=("rebalance_id", "nunique"),
            mean_car_event_equal=("event_mean", "mean"),
        )
        .reset_index()
    )
    pooled = (
        usable.groupby(["action", "pressure_quintile"], sort=False)[car_col]
        .agg(median_car_pooled="median", mean_car_pooled="mean")
        .reset_index()
    )
    ordered = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    table = summary.merge(pooled, on=["action", "pressure_quintile"], how="left")
    table["pressure_quintile"] = pd.Categorical(table["pressure_quintile"], categories=ordered, ordered=True)
    return table.sort_values(["action", "pressure_quintile"]).reset_index(drop=True)
