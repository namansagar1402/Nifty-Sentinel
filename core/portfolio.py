"""
User F&O Portfolio Manager & Real-Time Greeks Aggregator.
Tracks active Nifty option positions, live Mark-to-Market (MTM) PnL,
and Net Greeks risk exposure. Positions can be grouped into strategies and,
when a storage path is given, are saved to a JSON file after every change.
"""

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd

from core.models import Greeks, OptionType, StrategyLeg, TradeAction
from core.greeks import calculate_greeks

logger = logging.getLogger(__name__)

SHORT_MARGIN_PER_LOT_INR = 125000.0  # rough SPAN + exposure margin for one naked Nifty lot
DEFAULT_STOP_LOSS_PCT = 30.0
DEFAULT_TARGET_PCT = 50.0
NEAR_STOP_FRACTION = 0.8             # "near stop" once 80% of the stop distance is used


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
    group: Optional[str] = None


class PortfolioManager:
    """
    Manages active F&O positions, evaluates live MTM PnL, and calculates aggregate portfolio Greeks.
    """

    def __init__(self, storage_path: Optional[Union[str, Path]] = None):
        self.positions: List[PortfolioPosition] = []
        self.storage_path = Path(storage_path) if storage_path else None
        self.load_error: Optional[str] = None
        if self.storage_path and self.storage_path.exists():
            self._load()

    # ---- changes (each one is saved when a storage path is set) ----

    def add_position(
        self,
        strike: float,
        option_type: OptionType,
        action: TradeAction,
        quantity: int,
        entry_price: float,
        symbol: Optional[str] = None,
        entry_time: Optional[datetime] = None,
        group: Optional[str] = None,
        _save: bool = True,
    ) -> PortfolioPosition:
        """Adds a new option position to the user's active portfolio."""
        opt_str = option_type.value if hasattr(option_type, "value") else str(option_type)
        pos = PortfolioPosition(
            position_id=f"POS_{uuid.uuid4().hex[:12]}",
            symbol=symbol or f"NIFTY {int(strike)} {opt_str}",
            strike=float(strike),
            option_type=option_type,
            action=action,
            quantity=int(quantity),
            entry_price=float(entry_price),
            entry_time=entry_time or datetime.now(),
            current_price=float(entry_price),
            group=group,
        )
        self.positions.append(pos)
        if _save:
            self._save()
        return pos

    def add_strategy(self, name: str, legs: Sequence[StrategyLeg]) -> str:
        """Adds every leg of a strategy under one group name ("<name> #n") and returns that name."""
        existing = {p.group for p in self.positions if p.group}
        number = 1
        while f"{name} #{number}" in existing:
            number += 1
        group = f"{name} #{number}"
        for leg in legs:
            self.add_position(leg.strike, leg.option_type, leg.action, leg.quantity, leg.premium,
                              symbol=f"NIFTY {int(leg.strike)} {leg.option_type.value}", group=group, _save=False)
        self._save()
        return group

    def remove_position(self, position_id: str) -> bool:
        """Removes a position by its ID."""
        initial_len = len(self.positions)
        self.positions = [p for p in self.positions if p.position_id != position_id]
        self._save()
        return len(self.positions) < initial_len

    def remove_group(self, group: str) -> int:
        """Closes every leg of a strategy group; returns how many positions were removed."""
        initial_len = len(self.positions)
        self.positions = [p for p in self.positions if p.group != group]
        self._save()
        return initial_len - len(self.positions)

    def clear_portfolio(self):
        """Clears all positions."""
        self.positions = []
        self._save()

    # ---- saving ----

    def _save(self) -> None:
        if not self.storage_path:
            return
        records = [{
            "position_id": p.position_id, "symbol": p.symbol, "strike": p.strike,
            "option_type": p.option_type.value, "action": p.action.value, "quantity": p.quantity,
            "entry_price": p.entry_price, "entry_time": p.entry_time.isoformat(), "group": p.group,
        } for p in self.positions]
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.storage_path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"version": 1, "positions": records}, indent=2), encoding="utf-8")
            tmp.replace(self.storage_path)  # atomic: a crash never leaves half a file
        except OSError as exc:
            logger.error("Could not save portfolio to %s: %s", self.storage_path, exc)

    def _load(self) -> None:
        try:
            data = json.loads(self.storage_path.read_text(encoding="utf-8"))
            for r in data.get("positions", []):
                self.positions.append(PortfolioPosition(
                    position_id=r["position_id"], symbol=r["symbol"], strike=float(r["strike"]),
                    option_type=OptionType(r["option_type"]), action=TradeAction(r["action"]),
                    quantity=int(r["quantity"]), entry_price=float(r["entry_price"]),
                    entry_time=datetime.fromisoformat(r["entry_time"]),
                    current_price=float(r["entry_price"]), group=r.get("group"),
                ))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.positions = []
            self.load_error = f"Saved portfolio could not be read ({exc}); starting empty."
            logger.error(self.load_error)

    def to_csv(self, current_spot: float, india_vix: float = 14.85) -> str:
        """Current positions with live prices and P&L, as CSV text."""
        df = self.update_portfolio_mtm(current_spot, india_vix)["positions_dataframe"]
        return df.drop(columns=["position_id"], errors="ignore").to_csv(index=False)

    # ---- valuation ----

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

            if p.action == TradeAction.BUY:
                pos_pnl = (curr_px - p.entry_price) * p.quantity
                total_capital_deployed += p.entry_price * p.quantity
            else:
                pos_pnl = (p.entry_price - curr_px) * p.quantity
                total_capital_deployed += SHORT_MARGIN_PER_LOT_INR * (p.quantity / 50.0)

            pos_pnl_pct = ((curr_px - p.entry_price) / p.entry_price * 100.0) if p.entry_price > 0 else 0.0
            if p.action == TradeAction.SELL:
                pos_pnl_pct = -pos_pnl_pct

            p.mtm_pnl = round(pos_pnl, 2)
            p.mtm_pnl_pct = round(pos_pnl_pct, 2)
            total_mtm_pnl += pos_pnl

            net_delta += mult * g.delta * p.quantity
            net_gamma += mult * g.gamma * p.quantity
            net_theta += mult * g.theta * p.quantity  # ₹ decay per day
            net_vega += mult * g.vega * p.quantity    # ₹ impact per 1-point IV shift

            pos_records.append({
                "position_id": p.position_id,
                "group": p.group,
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


def greeks_in_words(mtm: Dict[str, Any], spot: float) -> Dict[str, Any]:
    """Turns net portfolio Greeks into ₹ figures and plain sentences."""
    move = spot * 0.01
    delta, gamma = mtm["net_delta"], mtm["net_gamma"]
    up = delta * move + 0.5 * gamma * move ** 2
    down = -delta * move + 0.5 * gamma * move ** 2
    theta, vega = mtm["net_theta_per_day_inr"], mtm["net_vega_per_1pct_inr"]

    def verb(x: float) -> str:
        return "gain" if x >= 0 else "lose"

    return {
        "move_up_1pct_inr": up,
        "move_down_1pct_inr": down,
        "theta_per_day_inr": theta,
        "vega_per_point_inr": vega,
        "delta_sentence": f"If Nifty rises 1% (about {move:,.0f} points) you {verb(up)} about ₹{abs(up):,.0f}; "
                          f"if it falls 1% you {verb(down)} about ₹{abs(down):,.0f}.",
        "theta_sentence": f"If nothing moves, you {verb(theta)} about ₹{abs(theta):,.0f} a day to time decay.",
        "vega_sentence": f"If India VIX rises 1 point you {verb(vega)} about ₹{abs(vega):,.0f}.",
    }


def position_alert(pnl_pct: float, stop_pct: float = DEFAULT_STOP_LOSS_PCT,
                   target_pct: float = DEFAULT_TARGET_PCT) -> str:
    """Status of one position against its stop-loss and target (both in % of entry premium)."""
    if pnl_pct <= -stop_pct:
        return "Stop hit"
    if pnl_pct >= target_pct:
        return "Target hit"
    if pnl_pct <= -stop_pct * NEAR_STOP_FRACTION:
        return "Near stop"
    return "OK"
