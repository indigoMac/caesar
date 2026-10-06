"""Matplotlib figures for the event study. No seaborn."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_car_paths(paths: pd.DataFrame, path: Path, *, xlabel: str, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if paths.empty:
        ax.text(0.5, 0.5, "No complete events in this window", ha="center", va="center")
        ax.set_axis_off()
    else:
        for action, group in paths.groupby("action", sort=False):
            ordered = group.sort_values("offset")
            ax.plot(ordered["offset"], ordered["mean_car"], marker="o", label=str(action))
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Event-equal-weighted mean CAR")
        ax.legend(frameon=False)
        ax.set_title(title)
    fig.tight_layout()
    return _save(fig, path)


def plot_action_means(summary: pd.DataFrame, path: Path, *, window: str) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    subset = summary[summary["window"] == window] if "window" in summary.columns else summary
    if subset.empty:
        ax.text(0.5, 0.5, "No window summary", ha="center", va="center")
        ax.set_axis_off()
    else:
        ax.bar(subset["action"].astype(str), subset["mean_car_event_equal"])
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_ylabel("Event-equal-weighted mean CAR")
        ax.set_title(f"Additions vs deletions: {window}")
    fig.tight_layout()
    return _save(fig, path)


def plot_quintiles(table: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if table.empty:
        ax.text(0.5, 0.5, "Fewer than five scored names per rebalance and action", ha="center", va="center")
        ax.set_axis_off()
    else:
        actions = list(table["action"].unique())
        width = 0.35 if len(actions) > 1 else 0.6
        quintiles = ["Q1", "Q2", "Q3", "Q4", "Q5"]
        positions = list(range(len(quintiles)))
        for index, action in enumerate(actions):
            group = table[table["action"] == action].copy()
            group["pressure_quintile"] = group["pressure_quintile"].astype(str)
            group = group.set_index("pressure_quintile")
            heights = [float(group.loc[label, "mean_car_event_equal"]) if label in group.index else float("nan") for label in quintiles]
            offset = (index - (len(actions) - 1) / 2) * width
            ax.bar([pos + offset for pos in positions], heights, width=width, label=str(action))
        ax.set_xticks(positions, quintiles)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_ylabel("Event-equal-weighted mean CAR")
        ax.set_title("CAR by pressure quintile (Q1 low, Q5 high)")
        ax.legend(frameon=False)
    fig.tight_layout()
    return _save(fig, path)


def plot_by_year(table: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if table.empty:
        ax.text(0.5, 0.5, "No year buckets", ha="center", va="center")
        ax.set_axis_off()
    else:
        for action, group in table.groupby("action", sort=False):
            ordered = group.sort_values("year")
            ax.plot(ordered["year"], ordered["mean_car_event_equal"], marker="o", label=str(action))
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel("Year of effective close")
        ax.set_ylabel("Event-equal-weighted mean CAR")
        ax.set_title("Primary-window CAR by year")
        ax.legend(frameon=False)
    fig.tight_layout()
    return _save(fig, path)


def plot_gross_vs_net(table: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if table.empty:
        ax.text(0.5, 0.5, "No portfolio returns", ha="center", va="center")
        ax.set_axis_off()
    else:
        labels = [f"{row.mode}\n{row.cost_model}" for row in table.itertuples(index=False)]
        positions = list(range(len(table)))
        width = 0.35
        ax.bar([pos - width / 2 for pos in positions], table["mean_gross_return"], width=width, label="Gross")
        ax.bar([pos + width / 2 for pos in positions], table["mean_net_return"], width=width, label="Net of costs")
        ax.set_xticks(positions, labels, fontsize=8)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_ylabel("Mean rebalance return")
        ax.set_title("Gross vs cost-adjusted buy-and-hold returns")
        ax.legend(frameon=False)
    fig.tight_layout()
    return _save(fig, path)


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path
