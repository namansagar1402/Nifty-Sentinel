"""
User F&O Portfolio Manager & Real-Time Greeks Aggregator.
Tracks active Nifty option positions, live Mark-to-Market (MTM) PnL,
and Net Greeks risk exposure.
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

from core.models import OptionType, TradeAction, Greeks
from core.greeks import black_scholes_price, calculate_greeks


@dataclass
class PortfolioPosition:
    position_id: str
    symbol: str
    strike: float
    option_type: OptionType
    action: TradeAction
    quantity: int
    entry_price: float
    entry_time: datetime
    current_price: float = 0.0
    mtm_pnl: float = 0.0
    mtm_pnl_pct: float = 0.0
    greeks: Optional[Greeks] = None


class PortfolioManager:
    """
    Manages active F&O positions, evaluates live MTM PnL, and calculates aggregate portfolio Greeks.
    """

    def __init__(self):
        self.positions: List[PortfolioPosition] = []

    def add_position(
        self,
        strike: float,
        option_type: OptionType,
        action: TradeAction,
        quantity: int,
        entry_price: float,
        symbol: Optional[str] = None,
        entry_time: Optional[datetime] = None,
    ) -> PortfolioPosition:
        """Adds a new option position to the user's active portfolio."""
        opt_str = option_type.value if hasattr(option_type, "value") else str(option_type)
        sym = symbol or f"NIFTY {int(strike)} {opt_str}"
        pos_id = f"POS_{len(self.positions) + 1}_{int(datetime.now().timestamp())}"

        pos = PortfolioPosition(
            position_id=pos_id,
            symbol=sym,
            strike=strike,
            option_type=option_type,
            action=action,
            quantity=quantity,
            entry_price=entry_price,
            entry_time=entry_time or datetime.now(),
            current_price=entry_price,
            mtm_pnl=0.0,
            mtm_pnl_pct=0.0,
        )
        self.positions.append(pos)
        return pos

    def remove_position(self, position_id: str) -> bool:
        """Removes a position by its ID."""
        initial_len = len(self.positions)
        self.positions = [p for p in self.positions if p.position_id != position_id]
        return len(self.positions) < initial_len

    def clear_portfolio(self):
        """Clears all positions."""
        self.positions = []

    def update_portfolio_mtm(
        self,
        current_spot: float,
        india_vix: float = 14.85,
        days_to_expiry: int = 5,
        risk_free_rate: float = 0.07,
    ) -> Dict[str, Any]:
        """
        Updates live prices, MTM PnL, and computes aggregated Portfolio Greeks.
        """
        if not self.positions:
            return self._empty_portfolio()

        tte = max(0.001, days_to_expiry / 365.0)
        iv_dec = max(0.08, india_vix / 100.0)

        total_capital_deployed = 0.0
        total_mtm_pnl = 0.0
        net_delta = 0.0
        net_gamma = 0.0
        net_theta = 0.0
        net_vega = 0.0

        pos_records = []

        for p in self.positions:
            g = calculate_greeks(current_spot, p.strike, tte, risk_free_rate, iv_dec, p.option_type)
            curr_px = round(g.price, 2)
            p.current_price = curr_px
            p.greeks = g

            mult = 1.0 if p.action == TradeAction.BUY else -1.0
            
            # Position PnL
            if p.action == TradeAction.BUY:
                pos_pnl = (curr_px - p.entry_price) * p.quantity
                cost = p.entry_price * p.quantity
                total_capital_deployed += cost
            else:
                pos_pnl = (p.entry_price - curr_px) * p.quantity
                margin = 125000.0 * (p.quantity / 50.0)
                total_capital_deployed += margin

            pos_pnl_pct = ((curr_px - p.entry_price) / p.entry_price * 100.0) if p.entry_price > 0 else 0.0
            if p.action == TradeAction.SELL:
                pos_pnl_pct = -pos_pnl_pct

            p.mtm_pnl = round(pos_pnl, 2)
            p.mtm_pnl_pct = round(pos_pnl_pct, 2)
            total_mtm_pnl += pos_pnl

            # Aggregate Greeks
            net_delta += mult * g.delta * p.quantity
            net_gamma += mult * g.gamma * p.quantity
            net_theta += mult * g.theta * p.quantity  # ₹ decay per day
            net_vega += mult * g.vega * p.quantity    # ₹ impact per 1% VIX shift

            pos_records.append({
                "position_id": p.position_id,
                "symbol": p.symbol,
                "action": p.action.value,
                "qty": p.quantity,
                "entry_px": p.entry_price,
                "ltp": curr_px,
                "mtm_pnl": round(pos_pnl, 2),
                "pnl_pct": round(pos_pnl_pct, 2),
                "delta": round(mult * g.delta * p.quantity, 2),
                "theta_daily_inr": round(mult * g.theta * p.quantity, 2),
            })

        pos_df = pd.DataFrame(pos_records)

        # Portfolio Health Status
        if net_theta > 500.0:
            theta_status = "Positive Theta Harvester (Earning from time decay)"
        elif net_theta < -1500.0:
            theta_status = "Severe Negative Theta Bleed (Time decay rapidly eroding portfolio)"
        else:
            theta_status = "Balanced Theta Profile"

        if abs(net_delta) > 150.0:
            delta_bias = f"{'Strongly Bullish' if net_delta > 0 else 'Strongly Bearish'} Directional Exposure"
        else:
            delta_bias = "Delta Neutral / Moderate Directional Bias"

        return {
            "total_positions": len(self.positions),
            "total_capital_deployed": round(total_capital_deployed, 2),
            "total_mtm_pnl": round(total_mtm_pnl, 2),
            "total_mtm_pct": round((total_mtm_pnl / total_capital_deployed * 100.0), 2) if total_capital_deployed > 0 else 0.0,
            "net_delta": round(net_delta, 2),
            "net_gamma": round(net_gamma, 4),
            "net_theta_per_day_inr": round(net_theta, 2),
            "net_vega_per_1pct_inr": round(net_vega, 2),
            "theta_status": theta_status,
            "delta_bias": delta_bias,
            "positions_dataframe": pos_df,
        }

    def _empty_portfolio(self) -> Dict[str, Any]:
        return {
            "total_positions": 0,
            "total_capital_deployed": 0.0,
            "total_mtm_pnl": 0.0,
            "total_mtm_pct": 0.0,
            "net_delta": 0.0,
            "net_gamma": 0.0,
            "net_theta_per_day_inr": 0.0,
            "net_vega_per_1pct_inr": 0.0,
            "theta_status": "No active positions",
            "delta_bias": "Neutral",
            "positions_dataframe": pd.DataFrame(),
        }
