"""
Core data models and data structures for Nifty Sentinel.
Supports trade records, options contracts, market regimes, and risk diagnostics.
"""

from dataclasses import dataclass, field
from datetime import datetime, date, time
from enum import Enum
from typing import List, Optional, Dict, Any


class OptionType(str, Enum):
    CALL = "CE"
    PUT = "PE"
    FUTURES = "FUT"
    EQUITY = "EQ"


class TradeAction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class MarketTrend(str, Enum):
    STRONG_BULLISH = "STRONG_BULLISH"
    BULLISH = "BULLISH"
    RANGEBOUND = "RANGEBOUND"
    BEARISH = "BEARISH"
    STRONG_BEARISH = "STRONG_BEARISH"
    HIGH_VOLATILITY_CHOP = "HIGH_VOLATILITY_CHOP"


class VolatilityRegime(str, Enum):
    LOW_IV = "LOW_IV"          # India VIX < 12.5
    MODERATE_IV = "MODERATE_IV"  # India VIX 12.5 - 16.5
    HIGH_IV = "HIGH_IV"        # India VIX 16.5 - 22.0
    EXTREME_IV = "EXTREME_IV"    # India VIX > 22.0


class CircuitBreakerStatus(str, Enum):
    NORMAL_GREEN = "GREEN (Trading Allowed)"
    WARNING_AMBER = "AMBER (Sizing Reduced 50%)"
    LOCKED_RED = "RED (Trading Halted - Daily Limit Hit)"


@dataclass
class OptionContract:
    symbol: str
    underlying: str = "NIFTY"
    strike_price: float = 0.0
    option_type: OptionType = OptionType.CALL
    expiry_date: Optional[date] = None
    lot_size: int = 50  # Default Nifty lot size


@dataclass
class RawTrade:
    trade_id: str
    symbol: str
    trade_time: datetime
    action: TradeAction
    quantity: int
    price: float
    order_type: str = "NRML"
    brokerage_and_charges: float = 0.0
    segment: str = "NFO"
    option_contract: Optional[OptionContract] = None


@dataclass
class ClosedTrade:
    """Represents a matched round-trip trade (Entry + Exit)."""
    trade_id: str
    symbol: str
    underlying: str
    option_type: OptionType
    strike_price: float
    expiry_date: Optional[date]
    direction: str  # "LONG" or "SHORT"
    entry_time: datetime
    exit_time: datetime
    quantity: int
    entry_price: float
    exit_price: float
    gross_pnl: float
    total_charges: float
    net_pnl: float
    pnl_percent: float
    holding_duration_mins: float
    is_expiry_day: bool
    is_revenge_trade: bool = False
    is_overtrade: bool = False
    is_martingale: bool = False


@dataclass
class DailySummary:
    date: date
    trade_count: int
    winning_trades: int
    losing_trades: int
    gross_pnl: float
    total_charges: float
    net_pnl: float
    max_intraday_drawdown: float
    turnover: float
    peak_capital: float = 0.0


@dataclass
class Greeks:
    price: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    iv: float


@dataclass
class StrategyLeg:
    action: TradeAction
    option_type: OptionType
    strike: float
    expiry_date: Optional[date]
    quantity: int
    premium: float
    greeks: Optional[Greeks] = None


@dataclass
class OptionsStrategy:
    name: str
    category: str  # "BULLISH", "BEARISH", "NEUTRAL", "VOLATILITY"
    legs: List[StrategyLeg]
    max_profit: float
    max_loss: float
    risk_reward_ratio: float
    breakeven_points: List[float]
    net_premium: float  # >0 credit, <0 debit
    net_delta: float
    net_gamma: float
    net_theta: float
    net_vega: float
    recommended_entry_rule: str
    recommended_stop_loss_rule: str
    recommended_target_rule: str
    rationale: str
