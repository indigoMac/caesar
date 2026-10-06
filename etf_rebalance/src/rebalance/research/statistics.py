"""Inference that treats a reconstitution as one experiment.

Stock-level IID t-statistics are not implemented. The primary estimate
equally weights rebalance-level mean outcomes. The bootstrap resamples whole
rebalance ids. A secondary pooled mean uses standard errors clustered by
rebalance id.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from rebalance.errors import DataContractError

INFERENCE_NOTE = (
    "Primary estimates equally weight rebalance events. Bootstrap resamples "
    "draw whole rebalance ids. Intervals are not stock-level IID t-statistics."
)


@dataclass(frozen=True)
class BootstrapResult:
    estimate: float
    ci_low: float
    ci_high: float
    n_events: int
    n_resamples: int
    confidence_level: float
    note: str


@dataclass(frozen=True)
class TInterval:
    estimate: float
    ci_low: float
    ci_high: float
    standard_error: float
    degrees_of_freedom: int
    n_events: int


@dataclass(frozen=True)
class ClusterResult:
    available: bool
    estimate: float
    standard_error: float
    ci_low: float
    ci_high: float
    n_clusters: int
    n_observations: int
    note: str


def event_level_means(frame: pd.DataFrame, value_col: str, rebalance_col: str = "rebalance_id") -> pd.Series:
    usable = frame.dropna(subset=[value_col])
    if usable.empty:
        raise DataContractError(f"no non-missing values in {value_col}")
    return usable.groupby(rebalance_col, sort=False)[value_col].mean()


def draw_rebalance_ids(
    rebalance_ids: np.ndarray,
    n_resamples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw whole rebalance ids with replacement. Shape is (resamples, n_events)."""
    unique = pd.unique(np.asarray(rebalance_ids))
    if len(unique) == 0:
        raise DataContractError("no rebalance ids to resample")
    choices = rng.integers(0, len(unique), size=(n_resamples, len(unique)))
    return unique[choices]


def expand_rebalance_draw(frame: pd.DataFrame, drawn_ids: np.ndarray, rebalance_col: str = "rebalance_id") -> pd.DataFrame:
    """Repeat every row of each drawn rebalance. Partial stock samples are not produced."""
    parts: list[pd.DataFrame] = []
    for sequence, rebalance_id in enumerate(drawn_ids):
        block = frame.loc[frame[rebalance_col] == rebalance_id].copy()
        if block.empty:
            raise DataContractError(f"bootstrap draw referenced unknown rebalance_id {rebalance_id}")
        block["_bootstrap_seq"] = sequence
        parts.append(block)
    return pd.concat(parts, ignore_index=True)


def bootstrap_event_means(
    frame: pd.DataFrame,
    value_col: str,
    *,
    rebalance_col: str = "rebalance_id",
    n_resamples: int = 1000,
    seed: int = 42,
    confidence_level: float = 0.95,
) -> BootstrapResult:
    means = event_level_means(frame, value_col, rebalance_col)
    estimate = float(means.mean())
    n_events = int(len(means))
    if n_events < 2:
        return BootstrapResult(
            estimate=estimate,
            ci_low=float("nan"),
            ci_high=float("nan"),
            n_events=n_events,
            n_resamples=0,
            confidence_level=confidence_level,
            note="at least two rebalance events are required for a bootstrap interval",
        )
    rng = np.random.default_rng(seed)
    drawn = draw_rebalance_ids(means.index.to_numpy(), n_resamples, rng)
    values = means.to_numpy(dtype=float)
    positions = pd.Index(means.index)
    # Map drawn ids back to the event-mean vector. A drawn rebalance contributes
    # its full-sample mean, never a resample of the stocks inside it.
    indexes = positions.get_indexer(drawn.ravel()).reshape(drawn.shape)
    if (indexes < 0).any():
        raise DataContractError("bootstrap draw contained an unknown rebalance id")
    simulated = values[indexes].mean(axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    low, high = np.quantile(simulated, [alpha, 1.0 - alpha])
    return BootstrapResult(
        estimate=estimate,
        ci_low=float(low),
        ci_high=float(high),
        n_events=n_events,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        note=INFERENCE_NOTE,
    )


def event_level_t_interval(event_means: pd.Series | np.ndarray, confidence_level: float = 0.95) -> TInterval:
    """Student-t interval whose sample size is the number of rebalances."""
    values = np.asarray(event_means, dtype=float)
    values = values[~np.isnan(values)]
    n_events = int(values.size)
    if n_events == 0:
        raise DataContractError("event-level interval received no means")
    estimate = float(values.mean())
    degrees = n_events - 1
    if n_events < 2:
        return TInterval(estimate, float("nan"), float("nan"), float("nan"), degrees, n_events)
    standard_error = float(values.std(ddof=1) / np.sqrt(n_events))
    critical = float(stats.t.ppf((1.0 + confidence_level) / 2.0, degrees))
    return TInterval(
        estimate=estimate,
        ci_low=estimate - critical * standard_error,
        ci_high=estimate + critical * standard_error,
        standard_error=standard_error,
        degrees_of_freedom=degrees,
        n_events=n_events,
    )


def pooled_clustered_mean(
    frame: pd.DataFrame,
    value_col: str,
    cluster_col: str = "rebalance_id",
    confidence_level: float = 0.95,
) -> ClusterResult:
    """Pooled stock-level mean with errors clustered by rebalance.

    This is a secondary diagnostic. It does not equally weight rebalances, and
    it is not reported as the primary estimate.
    """
    usable = frame.dropna(subset=[value_col])
    n_clusters = int(usable[cluster_col].nunique()) if not usable.empty else 0
    if n_clusters < 2 or len(usable) < 2:
        return ClusterResult(
            available=False,
            estimate=float("nan"),
            standard_error=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            n_clusters=n_clusters,
            n_observations=int(len(usable)),
            note="clustered inference needs at least two rebalance clusters",
        )
    outcome = usable[value_col].to_numpy(dtype=float)
    design = np.ones((len(outcome), 1))
    groups = usable[cluster_col].to_numpy()
    fit = sm.OLS(outcome, design).fit(cov_type="cluster", cov_kwds={"groups": groups})
    interval = np.asarray(fit.conf_int(alpha=1.0 - confidence_level))
    return ClusterResult(
        available=True,
        estimate=float(fit.params[0]),
        standard_error=float(fit.bse[0]),
        ci_low=float(interval[0, 0]),
        ci_high=float(interval[0, 1]),
        n_clusters=n_clusters,
        n_observations=int(len(outcome)),
        note="secondary pooled mean; stocks are not the independent experiments",
    )
