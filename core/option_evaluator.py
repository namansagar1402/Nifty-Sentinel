"""
Option Health Evaluator & Pre-Trade Testing Sandbox for Nifty 50 F&O.
Analyzes individual option contracts to assign OPTIMAL, SPECULATIVE, or HAZARDOUS ratings,
and provides mathematical TAKE vs DROP trade recommendations.
"""

from typing import Dict, Any, List, Optional, Tuple
import math
import numpy as np
import pandas as pd

from core.models import OptionType, TradeAction, Greeks
from core.greeks import black_scholes_price, calculate_greeks


# Verdict thresholds (decay = % of the premium lost per day to time decay).
OPTIMAL_MAX_DECAY_PCT = 9.0        # an ITM option above this is not "optimal"
ATM_OPTIMAL_MAX_DECAY_PCT = 4.5    # ATM options must decay slower than this to be optimal
ATM_OPTIMAL_MIN_PROB_PCT = 50.0
SPECULATIVE_MAX_DECAY_PCT = 9.0    # ATM / near-OTM limit for "speculative"
NEAR_OTM_MAX_DISTANCE_PCT = 1.0
DECAY_TRAP_PCT = 30.0              # losing this much a day is hazardous, whatever the moneyness
DROP_DECAY_PCT = 10.0              # a bought option decaying faster than this is dropped
HEDGE_DECAY_PCT = 6.0              # between HEDGE and DROP: suggest a spread
MIN_TAKE_PROB_PCT = 45.0           # TAKE needs at least this chance of profit


class OptionEvaluator:
    """
    Evaluates individual option contracts and advises traders before taking positions.
    """

    @classmethod
    def evaluate_option_health(
        cls,
        spot_price: float,
        strike_price: float,
        option_type: OptionType,
        action: TradeAction = TradeAction.BUY,
        days_to_expiry: int = 5,
        india_vix: float = 14.85,
        market_premium: Optional[float] = None,
        risk_free_rate: float = 0.07,
    ) -> Dict[str, Any]:
        """
        Assesses option risk and assigns OPTIMAL / SPECULATIVE / HAZARDOUS rating with granular metrics.
        """
        tte = max(0.001, days_to_expiry / 365.0)
        iv_dec = max(0.08, india_vix / 100.0)

        # Greeks
        g = calculate_greeks(spot_price, strike_price, tte, risk_free_rate, iv_dec, option_type)
        premium = market_premium if (market_premium is not None and market_premium > 0) else g.price
        premium = max(0.5, premium)

        # Moneyness & Distance
        dist_pts = strike_price - spot_price if option_type == OptionType.CALL else spot_price - strike_price
        dist_pct = (dist_pts / spot_price) * 100.0

        if option_type == OptionType.CALL:
            is_itm = strike_price < spot_price - 25
            is_otm = strike_price > spot_price + 25
        else:
            is_itm = strike_price > spot_price + 25
            is_otm = strike_price < spot_price - 25
        is_atm = not (is_itm or is_otm)

        moneyness_tag = "ITM" if is_itm else ("OTM" if is_otm else "ATM")

        # Probability of Profit (Approximate based on Delta)
        if action == TradeAction.BUY:
            prob_of_profit_pct = abs(g.delta) * 100.0
        else:
            prob_of_profit_pct = (1.0 - abs(g.delta)) * 100.0

        # Theta Decay Burn Rate (% of option premium lost per day just from time passage)
        daily_theta_burn_pct = (abs(g.theta) / premium) * 100.0 if premium > 0 else 50.0
        hourly_decay_inr = abs(g.theta) / 6.25  # ~6.25 trading hours per day

        # Required Breakeven Move before expiry
        if action == TradeAction.BUY:
            breakeven_price = (strike_price + premium) if option_type == OptionType.CALL else (strike_price - premium)
            req_move_pts = abs(breakeven_price - spot_price)
            req_move_pct = (req_move_pts / spot_price) * 100.0
        else:
            breakeven_price = (strike_price + premium) if option_type == OptionType.CALL else (strike_price - premium)
            req_move_pts = premium
            req_move_pct = (req_move_pts / spot_price) * 100.0

        # Safety Classification Algorithm
        # SAFE / LOW RISK:
        # - Defined probability > 55%
        # - Daily Theta Burn < 4.5%
        # - ITM or Near-ATM with high delta (>0.45)
        # RISKY / SPECULATIVE:
        # - Daily Theta Burn 4.5% - 9.0%
        # - OTM within 1.0% distance
        # - PoP 35% - 55%
        # HAZARDOUS / DANGEROUS:
        # - Far OTM (>1.2% distance)
        # - Daily Theta Burn > 9.0% (rapid 0-DTE bleed)
        # - PoP < 30%
        # - High IV crush vulnerability

        if action == TradeAction.BUY:
            is_near_otm = is_otm and abs(dist_pct) <= NEAR_OTM_MAX_DISTANCE_PCT
            if daily_theta_burn_pct > DECAY_TRAP_PCT:
                health_rating, health_color, health_badge, risk_level = "HAZARDOUS", "#ef4444", "HAZARDOUS", "VERY HIGH"
                summary = (f"Decay trap: this {moneyness_tag} option loses {daily_theta_burn_pct:.1f}% of its value a day "
                           f"to time decay (₹{abs(g.theta):.1f}/day). Even a correct view can lose money before expiry.")
            elif ((is_itm and daily_theta_burn_pct <= OPTIMAL_MAX_DECAY_PCT)
                  or (is_atm and daily_theta_burn_pct <= ATM_OPTIMAL_MAX_DECAY_PCT
                      and prob_of_profit_pct >= ATM_OPTIMAL_MIN_PROB_PCT)):
                health_rating, health_color, health_badge, risk_level = "OPTIMAL", "#10b981", "OPTIMAL", "LOW"
                summary = f"This {moneyness_tag} option has strong intrinsic value / Delta ({abs(g.delta):.2f}) and a manageable daily decay rate ({daily_theta_burn_pct:.1f}%/day). Good directional vehicle."
            elif is_itm:
                health_rating, health_color, health_badge, risk_level = "SPECULATIVE", "#f59e0b", "SPECULATIVE", "MODERATE"
                summary = (f"ITM with strong delta ({abs(g.delta):.2f}), but time decay is heavy: {daily_theta_burn_pct:.1f}% of "
                           f"the premium a day. Hold it only for a quick move.")
            elif (is_atm or is_near_otm) and daily_theta_burn_pct <= SPECULATIVE_MAX_DECAY_PCT:
                health_rating, health_color, health_badge, risk_level = "SPECULATIVE", "#f59e0b", "SPECULATIVE", "MODERATE"
                summary = f"ATM/OTM option with moderate delta ({abs(g.delta):.2f}) losing ₹{abs(g.theta):.1f}/day ({daily_theta_burn_pct:.1f}% of premium). Requires swift directional move within 24-48 hrs."
            else:
                health_rating, health_color, health_badge, risk_level = "HAZARDOUS", "#ef4444", "HAZARDOUS", "VERY HIGH"
                where = f"Far OTM option ({dist_pct:+.1f}% away)" if is_otm and not is_near_otm else f"{moneyness_tag} option"
                summary = f"Dangerous {where}. Fast decay ({daily_theta_burn_pct:.1f}%/day) with low probability of profit ({prob_of_profit_pct:.1f}%). Highly prone to expiring worthless."
        else:
            # Short / Option Writing Perspective
            if is_otm and abs(dist_pct) >= 1.2 and days_to_expiry <= 5:
                health_rating = "OPTIMAL"
                health_color = "#10b981"
                health_badge = "OPTIMAL"
                summary = f"Far OTM option with {prob_of_profit_pct:.1f}% probability of expiring worthless. Strong theta collection (₹{abs(g.theta):.1f}/day)."
                risk_level = "LOW (if hedged)"
            elif is_atm:
                health_rating = "SPECULATIVE"
                health_color = "#f59e0b"
                health_badge = "SPECULATIVE"
                summary = f"ATM Short option earns high theta (₹{abs(g.theta):.1f}/day) but carries heavy Gamma risk if spot breaks out."
                risk_level = "MODERATE"
            else:
                health_rating = "HAZARDOUS"
                health_color = "#ef4444"
                health_badge = "HAZARDOUS"
                summary = f"ITM Short option carries deep delta and unlimited loss liability without a hedge."
                risk_level = "VERY HIGH"

        return {
            "health_rating": health_rating,
            "health_badge": health_badge,
            "health_color": health_color,
            "risk_level": risk_level,
            "summary": summary,
            "strike": strike_price,
            "option_type": option_type.value,
            "action": action.value,
            "moneyness_tag": moneyness_tag,
            "distance_from_spot_pts": round(dist_pts, 1),
            "distance_from_spot_pct": round(dist_pct, 2),
            "premium": round(premium, 2),
            "delta": round(g.delta, 3),
            "gamma": round(g.gamma, 5),
            "theta_daily_inr": round(g.theta, 2),
            "theta_hourly_inr": round(hourly_decay_inr, 2),
            "daily_theta_burn_pct": round(daily_theta_burn_pct, 1),
            "vega_1pct": round(g.vega, 2),
            "iv_percent": round(iv_dec * 100.0, 1),
            "prob_of_profit_pct": round(prob_of_profit_pct, 1),
            "breakeven_price": round(breakeven_price, 1),
            "required_move_pct": round(req_move_pct, 2),
            "lot_size": 50,
            "capital_required_per_lot": round(premium * 50.0, 2) if action == TradeAction.BUY else 125000.0,
        }

    PNL_GRID_MOVES_PCT = (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0)
    PNL_GRID_MAX_DAY_COLUMNS = 6

    @classmethod
    def pnl_grid(
        cls,
        spot_price: float,
        strike_price: float,
        option_type: OptionType,
        action: TradeAction,
        days_to_expiry: float,
        india_vix: float,
        entry_premium: float,
        quantity: int,
        risk_free_rate: float = 0.07,
    ) -> pd.DataFrame:
        """
        Profit or loss in ₹ for Nifty moves (rows, % from today) against time passing (columns:
        "Today", "+N days", ..., "Expiry"). Prices are Black-Scholes at today's IV; expiry is intrinsic value.
        """
        iv_dec = max(0.08, india_vix / 100.0)
        whole_days = max(0, int(math.ceil(days_to_expiry)) - 1)
        step = max(1, int(math.ceil(whole_days / max(1, cls.PNL_GRID_MAX_DAY_COLUMNS - 2))))
        day_points = [0] + list(range(step, whole_days + 1, step))
        columns = [("Today" if d == 0 else f"+{d} day{'s' if d > 1 else ''}", max(0.0, days_to_expiry - d)) for d in day_points]
        columns.append(("Expiry", 0.0))

        sign = 1.0 if action == TradeAction.BUY else -1.0
        rows = {}
        for move in cls.PNL_GRID_MOVES_PCT:
            target = spot_price * (1.0 + move / 100.0)
            row = {}
            for label, days_left in columns:
                if days_left <= 0:
                    value = max(0.0, target - strike_price) if option_type == OptionType.CALL else max(0.0, strike_price - target)
                else:
                    value = black_scholes_price(target, strike_price, days_left / 365.0, risk_free_rate, iv_dec, option_type)
                row[label] = sign * (value - entry_premium) * quantity
            rows[move] = row
        return pd.DataFrame.from_dict(rows, orient="index")[[label for label, _ in columns]]

    @classmethod
    def test_trade_scenarios(
        cls,
        spot_price: float,
        strike_price: float,
        option_type: OptionType,
        action: TradeAction = TradeAction.BUY,
        days_to_expiry: int = 5,
        india_vix: float = 14.85,
        entry_premium: Optional[float] = None,
        lot_size: int = 50,
        num_lots: int = 1,
    ) -> Dict[str, Any]:
        """
        Simulates pre-trade what-if scenarios across price moves and time decay,
        generating an unambiguous TAKE vs DROP recommendation.
        """
        health = cls.evaluate_option_health(
            spot_price=spot_price,
            strike_price=strike_price,
            option_type=option_type,
            action=action,
            days_to_expiry=days_to_expiry,
            india_vix=india_vix,
            market_premium=entry_premium,
        )

        entry_px = health["premium"]
        total_qty = lot_size * num_lots
        total_investment = entry_px * total_qty

        # Scenario Matrix: Spot moves (-2%, -1%, -0.5%, 0%, +0.5%, +1%, +2%)
        spot_moves_pct = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
        time_horizons = [
            ("Today (T+0)", max(0.001, days_to_expiry / 365.0)),
            ("Tomorrow (T+1)", max(0.001, (days_to_expiry - 1) / 365.0)),
            ("Expiry (T=0)", 0.0),
        ]

        scenario_rows = []
        iv_dec = health["iv_percent"] / 100.0

        for move in spot_moves_pct:
            target_spot = spot_price * (1.0 + move / 100.0)
            row_dict = {
                "nifty_move_pct": f"{move:+.1f}%",
                "target_nifty_spot": round(target_spot, 1),
            }

            for t_name, tte_val in time_horizons:
                theo_px = black_scholes_price(target_spot, strike_price, tte_val, 0.07, iv_dec, option_type)
                
                if action == TradeAction.BUY:
                    pnl_per_share = theo_px - entry_px
                else:
                    pnl_per_share = entry_px - theo_px

                total_pnl = pnl_per_share * total_qty
                pnl_pct = (pnl_per_share / entry_px * 100.0) if entry_px > 0 else 0.0

                col_name = f"{t_name} PnL (₹)"
                row_dict[col_name] = round(total_pnl, 2)
                row_dict[f"{t_name} %"] = round(pnl_pct, 1)

            scenario_rows.append(row_dict)

        scenario_df = pd.DataFrame(scenario_rows)

        # Mathematical TAKE vs DROP Decision Logic
        # Criteria:
        # If UNSTABLE rating -> DROP OPTION
        # If BUY and Daily Theta Burn > 8% -> DROP OPTION (Decay Trap)
        # If BUY and Distance > 1.5% -> DROP OPTION (Lottery Ticket)
        # If BUY and SAFE/RISKY with PoP >= 45% -> TAKE TRADE
        # If BUY and Directional but High Theta -> CONVERT TO SPREAD

        if action == TradeAction.BUY:
            burn, pop = health["daily_theta_burn_pct"], health["prob_of_profit_pct"]
            if health["health_rating"] == "HAZARDOUS" or burn > DROP_DECAY_PCT or pop < MIN_TAKE_PROB_PCT:
                decision = "DROP"
                decision_badge = "DROP"
                decision_color = "#ef4444"
                if burn > DROP_DECAY_PCT:
                    decision_reason = f"Severe time decay: {burn:.1f}% of the premium lost every day (chance of profit {pop:.1f}%). The math is heavily against buyers here."
                elif pop < MIN_TAKE_PROB_PCT:
                    decision_reason = f"Only a {pop:.1f}% chance of profit, below the {MIN_TAKE_PROB_PCT:.0f}% this system requires before taking a trade."
                else:
                    decision_reason = health["summary"]
                suggested_alternative = f"Look at an In-The-Money strike ({spot_price - 100 if option_type == OptionType.CALL else spot_price + 100:.0f}), a later expiry, or a Defined-Risk Spread."
            elif burn >= HEDGE_DECAY_PCT:
                decision = "HEDGE"
                decision_badge = "HEDGE"
                decision_color = "#3b82f6"
                decision_reason = f"Option has good potential but naked holding risks losing ₹{abs(health['theta_daily_inr']) * total_qty:,.0f}/day to decay. Selling an OTM wing will finance the position."
                hedge_strike = strike_price + 200 if option_type == OptionType.CALL else strike_price - 200
                suggested_alternative = f"Buy Nifty {strike_price:.0f} {option_type.value} & Sell Nifty {hedge_strike:.0f} {option_type.value} to cap theta decay."
            else:
                decision = "TAKE"
                decision_badge = "TAKE"
                decision_color = "#10b981"
                decision_reason = f"High probability of profit ({health['prob_of_profit_pct']:.1f}%), low theta burn ({health['daily_theta_burn_pct']:.1f}%/day), and solid Delta leverage ({health['delta']:.2f})."
                suggested_alternative = f"Set Stop-Loss at ₹{entry_px * 0.70:.1f} (30% max loss) and Target at ₹{entry_px * 1.50:.1f}."
        else:
            # Short Option Decision
            if health["health_rating"] == "HAZARDOUS":
                decision = "DROP"
                decision_badge = "DROP"
                decision_color = "#ef4444"
                decision_reason = "Unhedged short position carries severe overnight gap liability."
                suggested_alternative = "Add a far OTM wing long option to form a Credit Spread."
            else:
                decision = "TAKE"
                decision_badge = "TAKE"
                decision_color = "#10b981"
                decision_reason = f"Collects ₹{abs(health['theta_daily_inr']) * total_qty:,.0f}/day with {health['prob_of_profit_pct']:.1f}% win probability."
                suggested_alternative = "Always maintain a protective wing or strict stop loss."

        return {
            "health": health,
            "decision": decision,
            "decision_badge": decision_badge,
            "decision_color": decision_color,
            "decision_reason": decision_reason,
            "suggested_alternative": suggested_alternative,
            "total_investment_inr": round(total_investment, 2),
            "total_quantity": total_qty,
            "scenario_table": scenario_df,
        }
