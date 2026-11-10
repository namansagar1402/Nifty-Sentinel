"""
Multi-leg Options Payoff & Greeks Engine for Nifty 50 strategies.
Computes breakeven points, max profit, max loss, risk-reward ratio, and net portfolio Greeks.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from core.models import StrategyLeg, TradeAction, OptionType, OptionsStrategy, Greeks
from core.greeks import calculate_greeks


MAX_REASONABLE_IV = 3.0  # 300% annualised; anything above is a units mistake


class PayoffCalculator:
    """
    Computes mathematical payoff matrices, breakeven boundaries, and Greeks for structured option strategies.
    """

    @classmethod
    def calculate_strategy_profile(
        cls,
        legs: List[StrategyLeg],
        nifty_spot: float = 24500.0,
        days_to_expiry: int = 5,
        risk_free_rate: float = 0.07,
        iv: float = 0.15,
        lot_size: int = 50,
    ) -> Dict[str, Any]:
        """
        Calculates complete payoff curve, Max Profit, Max Loss, and Breakevens for a multi-leg strategy.
        """
        if not legs:
            return cls._empty_profile()

        # IV is a decimal (0.17 = 17%). A value like India VIX 17.42 passed raw
        # would price options at 1,742% volatility and blow up the T+0 curve.
        if not 0.0 < iv <= MAX_REASONABLE_IV:
            raise ValueError(f"iv must be a decimal volatility (e.g. 0.17), got {iv}")

        tte = max(0.001, days_to_expiry / 365.0)

        # 1. Net Initial Premium (+ for credit, - for debit)
        net_premium_per_share = 0.0
        net_premium_total = 0.0
        for leg in legs:
            mult = -1.0 if leg.action == TradeAction.BUY else 1.0
            net_premium_per_share += mult * leg.premium
            net_premium_total += mult * leg.premium * leg.quantity

        # 2. Net Greeks
        net_delta = 0.0
        net_gamma = 0.0
        net_theta = 0.0
        net_vega = 0.0

        for leg in legs:
            mult = 1.0 if leg.action == TradeAction.BUY else -1.0
            g = calculate_greeks(nifty_spot, leg.strike, tte, risk_free_rate, iv, leg.option_type)
            net_delta += mult * g.delta * leg.quantity
            net_gamma += mult * g.gamma * leg.quantity
            net_theta += mult * g.theta * leg.quantity  # Daily decay in ₹
            net_vega += mult * g.vega * leg.quantity    # ₹ move per 1% IV change

        # 3. Payoff calculation across spot price range [spot - 6%, spot + 6%]
        price_range = np.linspace(nifty_spot * 0.94, nifty_spot * 1.06, 121)
        expiry_payoffs = []
        t0_payoffs = []

        for p in price_range:
            # Expiry Payoff (Intrinsic value at T=0)
            payoff_at_p = 0.0
            t0_val_at_p = 0.0

            for leg in legs:
                mult = 1.0 if leg.action == TradeAction.BUY else -1.0
                if leg.option_type == OptionType.CALL:
                    intrinsic = max(0.0, p - leg.strike)
                else:
                    intrinsic = max(0.0, leg.strike - p)

                payoff_at_p += mult * (intrinsic - leg.premium) * leg.quantity

                # T+0 theoretical price using Black Scholes
                t0_greek = calculate_greeks(p, leg.strike, tte, risk_free_rate, iv, leg.option_type)
                t0_val_at_p += mult * (t0_greek.price - leg.premium) * leg.quantity

            expiry_payoffs.append(payoff_at_p)
            t0_payoffs.append(t0_val_at_p)

        expiry_payoffs = np.array(expiry_payoffs)
        max_profit = float(np.max(expiry_payoffs))
        max_loss = abs(float(np.min(expiry_payoffs)))

        # Find Breakevens (zero crossings of expiry payoff)
        breakevens = []
        for i in range(len(price_range) - 1):
            y1, y2 = expiry_payoffs[i], expiry_payoffs[i + 1]
            if (y1 <= 0 and y2 >= 0) or (y1 >= 0 and y2 <= 0):
                # Linear interpolation for zero crossing
                if abs(y2 - y1) > 1e-6:
                    x_zero = price_range[i] + (0 - y1) * (price_range[i + 1] - price_range[i]) / (y2 - y1)
                    breakevens.append(round(float(x_zero), 1))

        # Risk-to-Reward
        risk_reward_ratio = (max_profit / max_loss) if max_loss > 0 else (99.0 if max_profit > 0 else 0.0)

        # Margin Estimate (Approximated for Indian exchange SPAN + Exposure)
        is_all_defined_risk = max_loss < 50000.0
        margin_estimate = min(150000.0, max(25000.0, max_loss * 1.3)) if is_all_defined_risk else 135000.0

        return {
            "nifty_spot": nifty_spot,
            "lot_size": lot_size,
            "net_premium_per_share": round(net_premium_per_share, 2),
            "net_premium_total_inr": round(net_premium_total, 2),
            "is_credit": net_premium_total > 0,
            "max_profit_inr": round(max_profit, 2),
            "max_loss_inr": round(max_loss, 2),
            "risk_reward_ratio": round(risk_reward_ratio, 2),
            "breakeven_points": breakevens,
            "net_delta": round(net_delta, 2),
            "net_gamma": round(net_gamma, 4),
            "net_theta_per_day_inr": round(net_theta, 2),
            "net_vega_per_1pct_inr": round(net_vega, 2),
            "margin_estimate_inr": round(margin_estimate, 2),
            "price_range": [round(float(x), 1) for x in price_range],
            "expiry_payoff_curve": [round(float(y), 2) for y in expiry_payoffs],
            "t0_payoff_curve": [round(float(y), 2) for y in t0_payoffs],
        }

    @staticmethod
    def _empty_profile() -> Dict[str, Any]:
        return {
            "nifty_spot": 24500.0,
            "lot_size": 50,
            "net_premium_per_share": 0.0,
            "net_premium_total_inr": 0.0,
            "is_credit": False,
            "max_profit_inr": 0.0,
            "max_loss_inr": 0.0,
            "risk_reward_ratio": 0.0,
            "breakeven_points": [],
            "net_delta": 0.0,
            "net_gamma": 0.0,
            "net_theta_per_day_inr": 0.0,
            "net_vega_per_1pct_inr": 0.0,
            "margin_estimate_inr": 0.0,
            "price_range": [],
            "expiry_payoff_curve": [],
            "t0_payoff_curve": [],
        }
