"""Directional signals. Deletion sign is applied by position weight, not by rewriting returns."""

from __future__ import annotations

from rebalance.errors import DataContractError


def action_signal(action: str) -> int:
    if action == "ADD":
        return 1
    if action == "DELETE":
        return -1
    raise DataContractError(f"action {action} has no directional signal")
