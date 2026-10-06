"""Point-in-time announcement state.

`event_state_asof` filters to what was public at a timestamp and then keeps
the latest row per security. A later NONE removes the name from the active
state. It does not delete the earlier ADD from the announcement log.
"""

from __future__ import annotations

import pandas as pd

from rebalance.calendar import as_utc
from rebalance.errors import DataContractError

GROUP_KEYS = ("rebalance_id", "index_name", "security_id")


def event_state_asof(events: pd.DataFrame, timestamp: pd.Timestamp) -> pd.DataFrame:
    """Return the active ADD/DELETE list that was public at `timestamp`.

    Rows with `published_at` after `timestamp` are invisible. Within the
    visible rows, the latest `(published_at, version, retrieved_at)` wins for
    each security. If that winning action is NONE, the security is absent
    from the returned state. The caller's `events` frame is not modified.
    """
    if "published_at" not in events.columns or "action" not in events.columns:
        raise DataContractError("events must include published_at and action")
    asof = as_utc(timestamp)
    published = pd.to_datetime(events["published_at"], utc=True)
    visible = events.loc[published <= asof].copy()
    if visible.empty:
        return visible.reset_index(drop=True)
    visible["__published_utc"] = published.loc[visible.index]
    sort_columns = ["__published_utc", "version"]
    if "retrieved_at" in visible.columns:
        visible["__retrieved_utc"] = pd.to_datetime(visible["retrieved_at"], utc=True)
        sort_columns.append("__retrieved_utc")
    visible = visible.sort_values(sort_columns, kind="mergesort")
    latest = visible.groupby(list(GROUP_KEYS), sort=False).tail(1)
    active = latest.loc[latest["action"] != "NONE"].copy()
    helper_columns = [column for column in ("__published_utc", "__retrieved_utc") if column in active.columns]
    return active.drop(columns=helper_columns).reset_index(drop=True)
