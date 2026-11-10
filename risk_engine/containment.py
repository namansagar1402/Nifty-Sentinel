"""
Predictive Drawdown Containment & Risk Control Engine for Nifty F&O Options.
Enforces Intraday Circuit Breaker state machine, Fractional Kelly Position Sizing,
and Adaptive Daily Loss Limits.
"""

from typing import Dict, Any, List, Optional, Tuple
import math
from core.models import CircuitBreakerStatus


class DrawdownContainmentEngine:
    """
    Real-time & predictive risk control engine that prevents retail account blowups.
    """

    @classmethod
    def evaluate_circuit_breaker(
        cls,
        current_daily_pnl: float,
        daily_loss_limit_inr: float,
        consecutive_losses: int = 0,
        daily_trade_count: int = 0,
        max_daily_trades: int = 4,
    ) -> Dict[str, Any]:
        """
        Evaluates the real-time trading state and determines whether trading should be permitted,
        downsized, or locked down.
        """
        amber_threshold = -0.50 * daily_loss_limit_inr
        red_threshold = -1.00 * daily_loss_limit_inr

        if current_daily_pnl <= red_threshold or consecutive_losses >= 3 or daily_trade_count >= max_daily_trades:
            status = CircuitBreakerStatus.LOCKED_RED
            action = "HALT ALL TRADING"
            if current_daily_pnl <= red_threshold:
                reason = f"Daily Max Loss Limit breached (-₹{abs(current_daily_pnl):,.2f} >= -₹{daily_loss_limit_inr:,.2f})."
            elif consecutive_losses >= 3:
                reason = "3 consecutive losses triggered emotional protection lock."
            else:
                reason = f"Daily trade count limit ({daily_trade_count}/{max_daily_trades}) reached."
            allowed_size_multiplier = 0.0
            can_trade = False

        elif current_daily_pnl <= amber_threshold or consecutive_losses == 2:
            status = CircuitBreakerStatus.WARNING_AMBER
            action = "REDUCE SIZING BY 50%"
            reason = f"Intraday drawdown warning: Loss at ₹{abs(current_daily_pnl):,.2f} (50% of budget) or 2 loss streak."
            allowed_size_multiplier = 0.50
            can_trade = True

        else:
            status = CircuitBreakerStatus.NORMAL_GREEN
            action = "NORMAL TRADING PERMITTED"
            reason = "Risk parameters within safe boundaries."
            allowed_size_multiplier = 1.0
            can_trade = True

        remaining_risk_budget = max(0.0, daily_loss_limit_inr + current_daily_pnl)

        return {
            "status": status.value,
            "can_trade": can_trade,
            "action_required": action,
            "reason": reason,
            "allowed_size_multiplier": allowed_size_multiplier,
            "remaining_risk_budget_inr": round(remaining_risk_budget, 2),
            "current_daily_pnl": round(current_daily_pnl, 2),
            "daily_loss_limit_inr": round(daily_loss_limit_inr, 2),
            "daily_trade_count": daily_trade_count,
            "max_daily_trades": max_daily_trades,
            "consecutive_losses": consecutive_losses,
        }

    @classmethod
    def calculate_position_sizing(
        cls,
        account_capital: float,
        win_rate_pct: float,
        payoff_ratio: float,
        trade_risk_pct_cap: float = 2.0,  # Max 2% capital risked per trade
        nifty_spot: float = 24500.0,
        sl_points: float = 30.0,
        lot_size: int = 50,
        fractional_kelly: float = 0.25,  # Quarter Kelly recommended for options
    ) -> Dict[str, Any]:
        """
        Calculates optimal position sizing using Fractional Kelly Criterion & Strict Risk Budgeting.
        """
        p = win_rate_pct / 100.0
        q = 1.0 - p
        b = max(0.1, payoff_ratio)

        # Full Kelly fraction: f* = (p*b - q) / b
        raw_kelly = (p * b - q) / b
        positive_edge = raw_kelly > 0

        # Fractional Kelly (Quarter Kelly)
        adj_kelly = max(0.0, raw_kelly * fractional_kelly) if positive_edge else 0.0

        # Capital risked under Kelly
        kelly_risk_amount = account_capital * min(adj_kelly, (trade_risk_pct_cap / 100.0))

        # Risk per unit / per lot
        risk_per_share = sl_points  # in ₹ points
        risk_per_lot = risk_per_share * lot_size

        if risk_per_lot > 0:
            recommended_lots = int(kelly_risk_amount / risk_per_lot)
        else:
            recommended_lots = 1

        recommended_lots = max(1, recommended_lots) if positive_edge else 1
        actual_total_risk = recommended_lots * risk_per_lot

        return {
            "account_capital": account_capital,
            "win_rate_pct": win_rate_pct,
            "payoff_ratio": payoff_ratio,
            "has_positive_edge": positive_edge,
            "full_kelly_pct": round(max(0.0, raw_kelly * 100.0), 2),
            "quarter_kelly_pct": round(adj_kelly * 100.0, 2),
            "recommended_risk_inr": round(kelly_risk_amount, 2),
            "recommended_lots": recommended_lots,
            "total_quantity": recommended_lots * lot_size,
            "risk_per_lot_inr": round(risk_per_lot, 2),
            "actual_total_risk_inr": round(actual_total_risk, 2),
            "actual_risk_pct_of_capital": round((actual_total_risk / account_capital) * 100.0, 2),
        }

    @classmethod
    def calculate_adaptive_daily_limit(
        cls,
        account_capital: float,
        india_vix: float = 14.5,
        base_risk_pct: float = 1.5,
    ) -> float:
        """
        Dynamically adjusts the daily loss threshold according to India VIX volatility regime.
        - High VIX (>18): Tighter percentage stop (e.g. 1.0 - 1.2%) to protect from high-gamma spikes.
        - Normal VIX (12-16): Base percentage stop (1.5%).
        - Low VIX (<12): Standard 1.5 - 2.0%.
        """
        if india_vix > 20.0:
            multiplier = 0.70  # Reduce loss tolerance by 30% in high volatility
        elif india_vix > 16.0:
            multiplier = 0.85
        elif india_vix < 11.5:
            multiplier = 1.10
        else:
            multiplier = 1.0

        daily_limit_pct = base_risk_pct * multiplier
        daily_limit_inr = account_capital * (daily_limit_pct / 100.0)

        return round(daily_limit_inr, 2)
