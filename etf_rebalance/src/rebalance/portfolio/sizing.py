"""Position weights with explicit caps.

Achieved gross exposure can be below the request when a per-name or liquidity
cap binds. The shortfall is recorded. Weight is not redistributed through a
cap. Liquidity participation is applied only when both NAV and a participation
limit are supplied.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from rebalance.errors import DataContractError
from rebalance.portfolio.signals import action_signal

Mode = str


@dataclass
class SizingResult:
    weights: pd.Series
    requested_gross: float
    achieved_gross: float
    notes: list[str] = field(default_factory=list)
    liquidity_constraint: str = "inactive"


def size_positions(
    securities: pd.DataFrame,
    *,
    mode: str,
    scheme: str,
    max_name_weight: float,
    gross_exposure_cap: float,
    benchmark_security_id: str | None = None,
    portfolio_nav: float | None = None,
    max_adv_participation: float | None = None,
) -> SizingResult:
    if max_name_weight <= 0 or gross_exposure_cap <= 0:
        raise DataContractError("weight caps must be positive")
    if benchmark_security_id and benchmark_security_id in set(securities["security_id"].astype(str)):
        raise DataContractError("benchmark security cannot also be sized as a reconstitution constituent")
    if scheme not in {"equal", "pressure"}:
        raise DataContractError(f"unknown sizing scheme {scheme}")
    notes: list[str] = []
    longs, shorts = _candidates(securities, mode)
    if mode == "long_additions_hedged" and not benchmark_security_id:
        raise DataContractError("hedged mode requires benchmark_security_id")
    nav_set = portfolio_nav is not None
    participation_set = max_adv_participation is not None
    if nav_set != participation_set:
        raise DataContractError("portfolio_nav and max_adv_participation must be supplied together")
    liquidity = "inactive"
    if nav_set:
        if portfolio_nav is None or portfolio_nav <= 0 or max_adv_participation is None or max_adv_participation <= 0:
            raise DataContractError("NAV and ADV participation must be positive")
        liquidity = "active"

    if mode == "long_short":
        long_budget = gross_exposure_cap / 2.0
        short_budget = gross_exposure_cap / 2.0
    elif mode == "long_additions_hedged":
        long_budget = gross_exposure_cap / 2.0
        short_budget = 0.0
    elif mode == "long_additions":
        long_budget = gross_exposure_cap
        short_budget = 0.0
    elif mode == "short_deletions":
        long_budget = 0.0
        short_budget = gross_exposure_cap
    else:
        raise DataContractError(f"unknown portfolio mode {mode}")

    weights = pd.Series(dtype=float)
    if long_budget > 0:
        if longs.empty:
            notes.append("no_long_names")
        else:
            allocated = _allocate_side(
                longs,
                budget=long_budget,
                sign=1.0,
                scheme=scheme,
                max_name_weight=max_name_weight,
                portfolio_nav=portfolio_nav,
                max_adv_participation=max_adv_participation,
                notes=notes,
            )
            weights = pd.concat([weights, allocated])
    if short_budget > 0:
        if shorts.empty:
            notes.append("no_short_names")
        else:
            allocated = _allocate_side(
                shorts,
                budget=short_budget,
                sign=-1.0,
                scheme=scheme,
                max_name_weight=max_name_weight,
                portfolio_nav=portfolio_nav,
                max_adv_participation=max_adv_participation,
                notes=notes,
            )
            weights = pd.concat([weights, allocated])
    if mode == "long_additions_hedged" and not weights.empty:
        long_gross = float(weights.clip(lower=0.0).sum())
        weights.loc[benchmark_security_id] = -long_gross
        notes.append("etf_hedge_matches_long_gross")
    achieved = float(weights.abs().sum()) if not weights.empty else 0.0
    if achieved + 1e-12 < gross_exposure_cap and "no_long_names" not in notes and "no_short_names" not in notes:
        if any(note.startswith("per_name_cap_binds") or note.startswith("liquidity_cap_binds") for note in notes):
            notes.append("achieved_gross_below_request")
    return SizingResult(
        weights=weights,
        requested_gross=gross_exposure_cap,
        achieved_gross=achieved,
        notes=notes,
        liquidity_constraint=liquidity,
    )


def _candidates(securities: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "action" not in securities.columns or "security_id" not in securities.columns:
        raise DataContractError("sizing inputs require security_id and action")
    longs = securities[securities["action"] == "ADD"]
    shorts = securities[securities["action"] == "DELETE"]
    if mode == "long_additions":
        return longs, securities.iloc[0:0]
    if mode == "short_deletions":
        return securities.iloc[0:0], shorts
    if mode in {"long_short", "long_additions_hedged"}:
        if mode == "long_additions_hedged":
            return longs, securities.iloc[0:0]
        return longs, shorts
    raise DataContractError(f"unknown portfolio mode {mode}")


def _allocate_side(
    names: pd.DataFrame,
    *,
    budget: float,
    sign: float,
    scheme: str,
    max_name_weight: float,
    portfolio_nav: float | None,
    max_adv_participation: float | None,
    notes: list[str],
) -> pd.Series:
    frame = names.drop_duplicates("security_id").set_index("security_id")
    if scheme == "equal":
        scores = pd.Series(1.0, index=frame.index)
    else:
        if "pressure_score" not in frame.columns or frame["pressure_score"].isna().any():
            missing = frame.index[frame["pressure_score"].isna()].astype(str).tolist() if "pressure_score" in frame.columns else list(frame.index.astype(str))
            raise DataContractError(f"pressure sizing missing scores for {missing}")
        if (frame["pressure_score"] < 0).any():
            raise DataContractError("pressure scores used for sizing must be non-negative intensities")
        scores = frame["pressure_score"].astype(float)
    caps = pd.Series(max_name_weight, index=frame.index, dtype=float)
    if portfolio_nav is not None:
        if "adv20" not in frame.columns or frame["adv20"].isna().any() or (frame["adv20"] <= 0).any():
            raise DataContractError("liquidity cap requires a positive adv20 on every sized name")
        participation_cap = max_adv_participation * frame["adv20"].astype(float) / portfolio_nav
        capped = participation_cap < caps
        if bool(capped.any()):
            notes.append("liquidity_cap_binds")
        caps = pd.concat([caps, participation_cap], axis=1).min(axis=1)
    raw = _allocate_capped(scores, budget, caps)
    if (raw < scores / scores.sum() * budget - 1e-10).any() and (raw >= caps - 1e-12).any():
        notes.append(f"per_name_cap_binds:{'long' if sign > 0 else 'short'}")
    signed = raw * sign
    # Touch the signal helper so a non-tradable action cannot be sized.
    for action in frame["action"].unique():
        action_signal(str(action))
    return signed


def _allocate_capped(scores: pd.Series, budget: float, caps: pd.Series) -> pd.Series:
    if (scores < 0).any():
        raise DataContractError("allocation scores must be non-negative")
    if float(scores.sum()) <= 0:
        raise DataContractError("allocation scores sum to zero")
    weights = pd.Series(0.0, index=scores.index, dtype=float)
    remaining = set(scores.index)
    remaining_budget = float(budget)
    while remaining and remaining_budget > 1e-12:
        free = scores.loc[list(remaining)]
        proposal = free / float(free.sum()) * remaining_budget
        capped_names = [name for name in proposal.index if float(proposal.loc[name]) > float(caps.loc[name]) + 1e-12]
        if not capped_names:
            weights.loc[proposal.index] = proposal.astype(float)
            remaining_budget = 0.0
            break
        for name in capped_names:
            weights.loc[name] = float(caps.loc[name])
            remaining_budget -= float(caps.loc[name])
            remaining.remove(name)
        if remaining_budget < -1e-9:
            raise DataContractError("capped allocation exceeded the budget")
    return weights
