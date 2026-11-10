"""
Expected move: the range the options market implies for Nifty over the next few trading days.

India VIX is the market's annualised expected volatility (in %). Over N trading days the
one-standard-deviation move is  VIX / sqrt(252) * sqrt(N)  percent, and about 68% of
outcomes fall inside +/- one standard deviation (assuming roughly normal returns).
"""

import math
from dataclasses import dataclass

from scipy.stats import norm

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class ExpectedMove:
    reference_price: float
    move_pct: float
    move_points: float
    low: float
    high: float
    probability_pct: float
    trading_days: int

    def contains(self, price: float) -> bool:
        return self.low <= price <= self.high


def expected_move(reference_price: float, india_vix: float, trading_days: int = 1,
                  sigmas: float = 1.0) -> ExpectedMove:
    """Range around `reference_price` that India VIX implies over `trading_days`."""
    if reference_price <= 0:
        raise ValueError(f"reference_price must be positive, got {reference_price}")
    if india_vix <= 0 or trading_days <= 0 or sigmas <= 0:
        raise ValueError("india_vix, trading_days and sigmas must be positive")

    move_pct = india_vix / math.sqrt(TRADING_DAYS_PER_YEAR) * math.sqrt(trading_days) * sigmas
    move_points = reference_price * move_pct / 100.0
    probability_pct = (norm.cdf(sigmas) - norm.cdf(-sigmas)) * 100.0
    return ExpectedMove(
        reference_price=reference_price,
        move_pct=move_pct,
        move_points=move_points,
        low=reference_price - move_points,
        high=reference_price + move_points,
        probability_pct=probability_pct,
        trading_days=trading_days,
    )
