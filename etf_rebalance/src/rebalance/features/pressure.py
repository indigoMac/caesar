"""Index-demand pressure scores.

The research model, once weights and fund AUM exist, is:

    NetFlow_i = sum_j AUM_j * (new_weight_ij - old_weight_ij)
    Pressure_i = NetFlow_i / ADV20_i

`estimated_flow` is that net flow, signed so positive means indexer buying.
It is optional. When it is missing, the score is an explicitly labelled proxy.
Proxy values are never written into `estimated_flow`.

TODO: Replace proxies with point-in-time index weights and fund AUM.
Do not backfill `estimated_flow` from `market_cap` or `delta_weight`.
"""

from __future__ import annotations

import math

import pandas as pd

from rebalance.errors import DataContractError


def compute_pressure(features: pd.DataFrame) -> pd.DataFrame:
    if "action" not in features.columns or "adv20" not in features.columns:
        raise DataContractError("pressure inputs require action and adv20")
    scored = features.copy()
    kinds: list[str] = []
    proxies: list[bool] = []
    scores: list[float] = []
    for row in scored.to_dict(orient="records"):
        kind, is_proxy, score = _score_row(row)
        kinds.append(kind)
        proxies.append(is_proxy)
        scores.append(score)
    scored["pressure_kind"] = kinds
    scored["pressure_is_proxy"] = proxies
    scored["pressure_score"] = scores
    if "estimated_flow" in features.columns:
        scored["estimated_flow"] = features["estimated_flow"].to_numpy()
    return scored


def _score_row(row: dict[str, object]) -> tuple[str, bool, float]:
    adv = _finite(row.get("adv20"))
    flow = _finite(row.get("estimated_flow"))
    delta = _finite(row.get("delta_weight"))
    market_cap = _finite(row.get("market_cap"))
    if adv is None or adv <= 0:
        return "unavailable", True, math.nan
    if flow is not None:
        sign = 1.0 if row["action"] == "ADD" else -1.0
        if row["action"] not in {"ADD", "DELETE"}:
            raise DataContractError(f"cannot score pressure for action {row['action']}")
        # Directional intensity. Positive means flow agrees with the event
        # (buying an addition, selling a deletion). This is not a proxy.
        return "net_flow_over_adv20", False, sign * flow / adv
    if delta is not None:
        return "abs_delta_weight_over_adv20", True, abs(delta) / adv
    if market_cap is not None:
        return "market_cap_over_adv20", True, market_cap / adv
    return "unavailable", True, math.nan


def _finite(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    number = float(value)  # type: ignore[arg-type]
    if math.isnan(number) or math.isinf(number):
        return None
    return number
