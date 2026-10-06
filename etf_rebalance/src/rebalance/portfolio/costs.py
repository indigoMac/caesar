"""Transaction-cost scenarios.

Net strategy return is gross return minus an entry cost and an exit cost.
Costs are positive fractions of traded notional. Zero is not the default.
A later spread, impact, or borrow model can implement `one_way_cost` without
changing the simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from rebalance.errors import DataContractError


def bps_to_fraction(bps: float) -> float:
    if bps < 0:
        raise DataContractError("transaction cost in bps cannot be negative")
    return float(bps) / 10_000.0


def net_return(gross_return: float, *, entry_cost: float, exit_cost: float) -> float:
    """Subtract one entry cost and one exit cost from a gross return."""
    if entry_cost < 0 or exit_cost < 0:
        raise DataContractError("entry and exit costs cannot be negative")
    return float(gross_return) - float(entry_cost) - float(exit_cost)


@dataclass(frozen=True)
class CostContext:
    security_id: str
    leg: Literal["entry", "exit"]
    adv20: float | None = None
    participation: float | None = None


class CostModel(Protocol):
    name: str

    def one_way_cost(self, context: CostContext) -> float:
        """Return a one-way cost as a fraction of notional."""


@dataclass(frozen=True)
class FlatBpsCostModel:
    """Constant one-way cost. The same fraction is charged on entry and exit."""

    bps: float
    name: str = "flat_bps"

    def __post_init__(self) -> None:
        if self.bps < 0:
            raise DataContractError("flat cost bps cannot be negative")

    def one_way_cost(self, context: CostContext) -> float:
        del context
        return bps_to_fraction(self.bps)
