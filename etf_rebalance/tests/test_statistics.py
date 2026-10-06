"""Rebalance-level inference. Stock rows inside an event are not resampled alone."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rebalance.research.cross_section import assign_pressure_quintiles
from rebalance.research.statistics import (
    bootstrap_event_means,
    draw_rebalance_ids,
    event_level_means,
    event_level_t_interval,
    expand_rebalance_draw,
    pooled_clustered_mean,
)


def _panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rebalance_id": ["A", "A", "A", "B", "B", "C", "C"],
            "security_id": ["1", "2", "3", "4", "5", "6", "7"],
            "car": [1.0, 1.0, 100.0, 0.0, 0.0, 2.0, 2.0],
        }
    )


def test_bootstrap_resamples_whole_rebalance_ids() -> None:
    frame = _panel()
    means = event_level_means(frame, "car")
    assert means.loc["A"] == 34.0
    assert means.loc["B"] == 0.0
    result = bootstrap_event_means(frame, "car", n_resamples=200, seed=7)
    assert result.estimate == (34.0 + 0.0 + 2.0) / 3.0
    assert result.n_events == 3
    repeated = bootstrap_event_means(frame, "car", n_resamples=200, seed=7)
    assert result.ci_low == repeated.ci_low
    assert result.ci_high == repeated.ci_high

    rng = np.random.default_rng(1)
    drawn = draw_rebalance_ids(frame["rebalance_id"].to_numpy(), n_resamples=20, rng=rng)
    assert set(np.unique(drawn)) <= {"A", "B", "C"}
    expanded = expand_rebalance_draw(frame, np.array(["A", "A"]))
    assert len(expanded) == 6
    assert set(expanded.loc[expanded["_bootstrap_seq"] == 0, "security_id"]) == {"1", "2", "3"}
    assert list(expanded.loc[expanded["_bootstrap_seq"] == 0, "car"]) == [1.0, 1.0, 100.0]


def test_t_interval_uses_rebalance_degrees_of_freedom() -> None:
    means = event_level_means(_panel(), "car")
    interval = event_level_t_interval(means)
    assert interval.n_events == 3
    assert interval.degrees_of_freedom == 2
    assert interval.degrees_of_freedom != len(_panel()) - 1


def test_clustered_mean_counts_rebalances_as_clusters() -> None:
    clustered = pooled_clustered_mean(_panel(), "car")
    assert clustered.available
    assert clustered.n_clusters == 3
    assert clustered.n_observations == 7


def test_quintiles_are_assigned_inside_each_rebalance() -> None:
    frame = pd.DataFrame(
        {
            "rebalance_id": ["A"] * 5 + ["B"] * 5 + ["C"] * 4,
            "action": ["ADD"] * 14,
            "pressure_score": [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 1, 2, 3, 4],
        }
    )
    ranked = assign_pressure_quintiles(frame)
    group_a = ranked[ranked["rebalance_id"] == "A"]
    group_b = ranked[ranked["rebalance_id"] == "B"]
    group_c = ranked[ranked["rebalance_id"] == "C"]
    assert group_a.loc[group_a["pressure_score"] == 5, "pressure_quintile"].iloc[0] == "Q5"
    assert group_a.loc[group_a["pressure_score"] == 1, "pressure_quintile"].iloc[0] == "Q1"
    assert group_b.loc[group_b["pressure_score"] == 10, "pressure_quintile"].iloc[0] == "Q1"
    assert (group_c["quintile_status"] == "insufficient_names").all()
