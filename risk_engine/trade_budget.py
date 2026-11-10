"""Does a single trade fit inside today's daily loss limit?"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TradeBudget:
    max_loss_inr: float
    daily_limit_inr: float
    limit_used_ratio: float      # 1.0 = the trade alone could use the whole day's limit
    fits: bool
    max_lots_within_limit: int   # how many lots keep the worst case inside the limit


def check_trade_budget(max_loss_inr: float, daily_limit_inr: float, loss_per_lot_inr: float) -> TradeBudget:
    """Compares a trade's worst-case loss with the daily loss limit."""
    if daily_limit_inr <= 0:
        raise ValueError(f"daily_limit_inr must be positive, got {daily_limit_inr}")
    max_lots = math.floor(daily_limit_inr / loss_per_lot_inr) if loss_per_lot_inr > 0 else 0
    return TradeBudget(
        max_loss_inr=max_loss_inr,
        daily_limit_inr=daily_limit_inr,
        limit_used_ratio=max_loss_inr / daily_limit_inr,
        fits=max_loss_inr <= daily_limit_inr,
        max_lots_within_limit=max_lots,
    )
