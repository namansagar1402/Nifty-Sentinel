"""
Optimal Defined-Risk Strategy Recommendation Engine for Nifty 50.
Recommends high-probability, risk-defined option structures (Spreads, Condors, Butterflies)
tailored to contain tail-risk drawdowns.
"""

from typing import List, Dict, Any, Optional
import math
from core.models import StrategyLeg, TradeAction, OptionType, OptionsStrategy
from strategy_optimizer.regime_classifier import MarketRegimeClassifier
from strategy_optimizer.payoff_calculator import PayoffCalculator
from core.greeks import black_scholes_price


class StrategyRecommender:
    """
    Synthesizes market regime metrics to recommend mathematically sound options strategies.
    """

    @classmethod
    def get_recommendations(
        cls,
        nifty_spot: float = 24500.0,
        india_vix: float = 14.5,
        iv_percentile: float = 45.0,
        ema_20: float = 24450.0,
        ema_50: float = 24300.0,
        pcr_oi: float = 1.05,
        max_pain: float = 24500.0,
        days_to_expiry: int = 5,
        lot_size: int = 50,
        include_all: bool = False,
    ) -> Dict[str, Any]:
        """
        Generates defined-risk strategies for the current market.

        By default only strategies that fit today's regime are returned; with include_all=True all six
        come back (fitting ones first), each with its real `legs` and a `fits_regime` flag.
        """
        # Round spot to nearest 50-strike
        atm_strike = round(nifty_spot / 50.0) * 50.0

        # Classify regime
        regime = MarketRegimeClassifier.classify_regime(
            nifty_spot=nifty_spot,
            ema_20=ema_20,
            ema_50=ema_50,
            india_vix=india_vix,
            iv_percentile=iv_percentile,
            pcr_oi=pcr_oi,
            max_pain_strike=max_pain,
        )

        trend_bias = regime["directional_bias"]
        vol_regime = regime["volatility_regime"]
        iv_dec = max(0.08, india_vix / 100.0)
        tte = max(0.001, days_to_expiry / 365.0)

        # Helper to price strikes
        def get_p(strike: float, o_type: OptionType) -> float:
            return round(black_scholes_price(nifty_spot, strike, tte, 0.07, iv_dec, o_type), 1)

        candidate_strategies = []

        # 1. Bullish Low/Moderate IV -> Bull Call Debit Spread
        fits_bull_call = trend_bias == "BULLISH" and vol_regime in ["LOW_IV", "MODERATE_IV"]
        if fits_bull_call or include_all:
            buy_strike = atm_strike
            sell_strike = atm_strike + 200.0
            p_buy = get_p(buy_strike, OptionType.CALL)
            p_sell = get_p(sell_strike, OptionType.CALL)

            legs = [
                StrategyLeg(TradeAction.BUY, OptionType.CALL, buy_strike, None, lot_size, p_buy),
                StrategyLeg(TradeAction.SELL, OptionType.CALL, sell_strike, None, lot_size, p_sell),
            ]
            profile = PayoffCalculator.calculate_strategy_profile(legs, nifty_spot, days_to_expiry, 0.07, iv_dec, lot_size)

            candidate_strategies.append({
                "strategy_name": "Bull Call Debit Spread",
                "legs": legs,
                "fits_regime": fits_bull_call,
                "suitability_rank": 1,
                "regime_fit": "Optimal for Bullish Trend with Cheaper Implied Volatility",
                "category": "BULLISH",
                "legs_detail": [
                    f"BUY Nifty {buy_strike:.0f} CE @ ₹{p_buy:.1f}",
                    f"SELL Nifty {sell_strike:.0f} CE @ ₹{p_sell:.1f}",
                ],
                "profile": profile,
                "entry_rule": f"Enter when Nifty breaks above {ema_20:.0f} with ADX > 20.",
                "stop_loss_rule": "Exit if total spread value loses 40% of initial debit, or Nifty closes below 50 EMA.",
                "target_rule": "Target 60% to 75% of maximum profit (₹{:.0f}).".format(profile["max_profit_inr"]),
                "containment_protection": "100% Defined Maximum Loss. Eliminates catastrophic overnight gap risk.",
            })

        # 2. Bullish High IV -> Bull Put Credit Spread
        fits_bull_put = trend_bias == "BULLISH" and vol_regime in ["HIGH_IV", "EXTREME_IV", "MODERATE_IV"]
        if fits_bull_put or include_all:
            sell_strike = atm_strike - 150.0
            buy_strike = sell_strike - 200.0
            p_sell = get_p(sell_strike, OptionType.PUT)
            p_buy = get_p(buy_strike, OptionType.PUT)

            legs = [
                StrategyLeg(TradeAction.SELL, OptionType.PUT, sell_strike, None, lot_size, p_sell),
                StrategyLeg(TradeAction.BUY, OptionType.PUT, buy_strike, None, lot_size, p_buy),
            ]
            profile = PayoffCalculator.calculate_strategy_profile(legs, nifty_spot, days_to_expiry, 0.07, iv_dec, lot_size)

            candidate_strategies.append({
                "strategy_name": "Bull Put Credit Spread (OTM)",
                "legs": legs,
                "fits_regime": fits_bull_put,
                "suitability_rank": 1 if vol_regime in ["HIGH_IV", "EXTREME_IV"] else 2,
                "regime_fit": "Optimal for Bullish / Moderate Bullish with High IV Crush opportunity",
                "category": "BULLISH",
                "legs_detail": [
                    f"SELL Nifty {sell_strike:.0f} PE @ ₹{p_sell:.1f} (OTM Put Write)",
                    f"BUY Nifty {buy_strike:.0f} PE @ ₹{p_buy:.1f} (Protective Wing)",
                ],
                "profile": profile,
                "entry_rule": f"Enter when Nifty holds support above {sell_strike:.0f} with PCR > 1.0.",
                "stop_loss_rule": f"Exit if Nifty spot breaches below short strike {sell_strike:.0f} or spread doubles.",
                "target_rule": "Target 70% of collected net credit.",
                "containment_protection": "Protective long put caps max loss at ₹{:.0f} if sudden black swan gap-down occurs.".format(profile["max_loss_inr"]),
            })

        # 3. Bearish Low/Moderate IV -> Bear Put Debit Spread
        fits_bear_put = trend_bias == "BEARISH" and vol_regime in ["LOW_IV", "MODERATE_IV"]
        if fits_bear_put or include_all:
            buy_strike = atm_strike
            sell_strike = atm_strike - 200.0
            p_buy = get_p(buy_strike, OptionType.PUT)
            p_sell = get_p(sell_strike, OptionType.PUT)

            legs = [
                StrategyLeg(TradeAction.BUY, OptionType.PUT, buy_strike, None, lot_size, p_buy),
                StrategyLeg(TradeAction.SELL, OptionType.PUT, sell_strike, None, lot_size, p_sell),
            ]
            profile = PayoffCalculator.calculate_strategy_profile(legs, nifty_spot, days_to_expiry, 0.07, iv_dec, lot_size)

            candidate_strategies.append({
                "strategy_name": "Bear Put Debit Spread",
                "legs": legs,
                "fits_regime": fits_bear_put,
                "suitability_rank": 1,
                "regime_fit": "Optimal for Downward Momentum with Low/Moderate IV",
                "category": "BEARISH",
                "legs_detail": [
                    f"BUY Nifty {buy_strike:.0f} PE @ ₹{p_buy:.1f}",
                    f"SELL Nifty {sell_strike:.0f} PE @ ₹{p_sell:.1f}",
                ],
                "profile": profile,
                "entry_rule": f"Enter on breakdown below {ema_20:.0f} with PCR < 0.85.",
                "stop_loss_rule": "Exit if spread loses 40% of net debit.",
                "target_rule": "Target 65% of max profit.",
                "containment_protection": "Max loss capped at initial net debit. No unlimited loss vulnerability.",
            })

        # 4. Bearish High IV -> Bear Call Credit Spread
        fits_bear_call = trend_bias == "BEARISH" and vol_regime in ["HIGH_IV", "EXTREME_IV", "MODERATE_IV"]
        if fits_bear_call or include_all:
            sell_strike = atm_strike + 150.0
            buy_strike = sell_strike + 200.0
            p_sell = get_p(sell_strike, OptionType.CALL)
            p_buy = get_p(buy_strike, OptionType.CALL)

            legs = [
                StrategyLeg(TradeAction.SELL, OptionType.CALL, sell_strike, None, lot_size, p_sell),
                StrategyLeg(TradeAction.BUY, OptionType.CALL, buy_strike, None, lot_size, p_buy),
            ]
            profile = PayoffCalculator.calculate_strategy_profile(legs, nifty_spot, days_to_expiry, 0.07, iv_dec, lot_size)

            candidate_strategies.append({
                "strategy_name": "Bear Call Credit Spread (OTM)",
                "legs": legs,
                "fits_regime": fits_bear_call,
                "suitability_rank": 1 if vol_regime in ["HIGH_IV", "EXTREME_IV"] else 2,
                "regime_fit": "Optimal for Bearish Resistance with High Option Premiums",
                "category": "BEARISH",
                "legs_detail": [
                    f"SELL Nifty {sell_strike:.0f} CE @ ₹{p_sell:.1f}",
                    f"BUY Nifty {buy_strike:.0f} CE @ ₹{p_buy:.1f}",
                ],
                "profile": profile,
                "entry_rule": f"Enter when Nifty faces resistance near {sell_strike:.0f} and VIX is elevated.",
                "stop_loss_rule": f"Exit if Nifty closes above {sell_strike:.0f}.",
                "target_rule": "Capture 70% of initial credit.",
                "containment_protection": "Protective long call caps worst-case loss at ₹{:.0f}.".format(profile["max_loss_inr"]),
            })

        # 5. Rangebound / Neutral High IV -> Iron Condor
        fits_neutral = trend_bias in ["NEUTRAL", "RANGEBOUND", "HIGH_VOLATILITY_CHOP"]
        if fits_neutral or include_all:
            # Iron Condor (Sell OTM Put + Buy Far OTM Put; Sell OTM Call + Buy Far OTM Call)
            put_sell = atm_strike - 250.0
            put_buy = put_sell - 200.0
            call_sell = atm_strike + 250.0
            call_buy = call_sell + 200.0

            p_put_sell = get_p(put_sell, OptionType.PUT)
            p_put_buy = get_p(put_buy, OptionType.PUT)
            p_call_sell = get_p(call_sell, OptionType.CALL)
            p_call_buy = get_p(call_buy, OptionType.CALL)

            legs = [
                StrategyLeg(TradeAction.SELL, OptionType.PUT, put_sell, None, lot_size, p_put_sell),
                StrategyLeg(TradeAction.BUY, OptionType.PUT, put_buy, None, lot_size, p_put_buy),
                StrategyLeg(TradeAction.SELL, OptionType.CALL, call_sell, None, lot_size, p_call_sell),
                StrategyLeg(TradeAction.BUY, OptionType.CALL, call_buy, None, lot_size, p_call_buy),
            ]
            profile = PayoffCalculator.calculate_strategy_profile(legs, nifty_spot, days_to_expiry, 0.07, iv_dec, lot_size)

            candidate_strategies.append({
                "strategy_name": "4-Leg Defined Iron Condor",
                "legs": legs,
                "fits_regime": fits_neutral,
                "suitability_rank": 1,
                "regime_fit": "Master Strategy for Rangebound Consolidation & Theta Decay Collection",
                "category": "NEUTRAL",
                "legs_detail": [
                    f"SELL Nifty {put_sell:.0f} PE @ ₹{p_put_sell:.1f}",
                    f"BUY Nifty {put_buy:.0f} PE @ ₹{p_put_buy:.1f}",
                    f"SELL Nifty {call_sell:.0f} CE @ ₹{p_call_sell:.1f}",
                    f"BUY Nifty {call_buy:.0f} CE @ ₹{p_call_buy:.1f}",
                ],
                "profile": profile,
                "entry_rule": f"Deploy on Monday/Tuesday when Nifty expected to oscillate between {put_sell:.0f} and {call_sell:.0f}.",
                "stop_loss_rule": f"Exit if Nifty spot approaches within 50 pts of either short strike ({put_sell:.0f} or {call_sell:.0f}).",
                "target_rule": "Exit when 50% - 60% of max profit is achieved (Theta harvest).",
                "containment_protection": "Both sides strictly wing-hedged. Zero margin call danger.",
            })

            # 6. Iron Butterfly (ATM Straddle + OTM Wings)
            ib_put_buy = atm_strike - 300.0
            ib_call_buy = atm_strike + 300.0
            p_atm_ce = get_p(atm_strike, OptionType.CALL)
            p_atm_pe = get_p(atm_strike, OptionType.PUT)
            p_ib_pe = get_p(ib_put_buy, OptionType.PUT)
            p_ib_ce = get_p(ib_call_buy, OptionType.CALL)

            ib_legs = [
                StrategyLeg(TradeAction.SELL, OptionType.CALL, atm_strike, None, lot_size, p_atm_ce),
                StrategyLeg(TradeAction.SELL, OptionType.PUT, atm_strike, None, lot_size, p_atm_pe),
                StrategyLeg(TradeAction.BUY, OptionType.CALL, ib_call_buy, None, lot_size, p_ib_ce),
                StrategyLeg(TradeAction.BUY, OptionType.PUT, ib_put_buy, None, lot_size, p_ib_pe),
            ]
            ib_profile = PayoffCalculator.calculate_strategy_profile(ib_legs, nifty_spot, days_to_expiry, 0.07, iv_dec, lot_size)

            candidate_strategies.append({
                "strategy_name": "Iron Butterfly (ATM Theta Pin)",
                "legs": ib_legs,
                "fits_regime": fits_neutral,
                "suitability_rank": 2,
                "regime_fit": "High Reward:Risk Pin Strategy when Nifty is pinned near Max Pain",
                "category": "NEUTRAL",
                "legs_detail": [
                    f"SELL Nifty {atm_strike:.0f} CE @ ₹{p_atm_ce:.1f}",
                    f"SELL Nifty {atm_strike:.0f} PE @ ₹{p_atm_pe:.1f}",
                    f"BUY Nifty {ib_call_buy:.0f} CE @ ₹{p_ib_ce:.1f}",
                    f"BUY Nifty {ib_put_buy:.0f} PE @ ₹{p_ib_pe:.1f}",
                ],
                "profile": ib_profile,
                "entry_rule": f"Enter when Nifty consolidates tightly around Max Pain strike ({max_pain:.0f}).",
                "stop_loss_rule": "Exit if loss reaches 1.0x initial credit collected.",
                "target_rule": "Take profit at 30% - 40% of max profit.",
                "containment_protection": "Wing hedges guarantee max risk cannot exceed ₹{:.0f}.".format(ib_profile["max_loss_inr"]),
            })

        # Strategies that fit today's regime first, then by suitability rank.
        candidate_strategies = sorted(candidate_strategies,
                                      key=lambda s: (not s["fits_regime"], s["suitability_rank"]))
        fitting = [s for s in candidate_strategies if s["fits_regime"]]

        return {
            "regime_summary": regime,
            "recommended_strategies": candidate_strategies,
            "top_strategy": fitting[0] if fitting else None,
        }
