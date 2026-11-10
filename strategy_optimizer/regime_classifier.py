"""
Nifty 50 Market Regime Classifier.
Analyzes Trend, India VIX, Implied Volatility Percentile (IVP), PCR (Put-Call Ratio), and Max Pain.
"""

from typing import Dict, Any, Optional
from core.models import MarketTrend, VolatilityRegime


class MarketRegimeClassifier:
    """
    Classifies the current market state into precise trend and volatility regimes.
    """

    @staticmethod
    def classify_regime(
        nifty_spot: float = 24500.0,
        ema_20: float = 24450.0,
        ema_50: float = 24300.0,
        india_vix: float = 14.5,
        iv_percentile: float = 45.0,  # 0 to 100
        pcr_oi: float = 1.05,        # Put-Call Ratio
        max_pain_strike: float = 24500.0,
        adx_14: float = 22.0,        # Trend strength (>25 = strong trend)
    ) -> Dict[str, Any]:
        """
        Computes composite market regime and recommended options bias.
        """
        # 1. Volatility Regime
        if india_vix < 12.5 or iv_percentile < 25.0:
            vol_regime = VolatilityRegime.LOW_IV
            vol_desc = "Low Volatility (Options Cheaper, Premium Buying / Debit Spreads favored)"
            premium_bias = "BUY_DEBIT_SPREADS"
        elif india_vix <= 16.5 or iv_percentile <= 65.0:
            vol_regime = VolatilityRegime.MODERATE_IV
            vol_desc = "Normal Volatility (Balanced Greeks, Credit or Debit Spreads suitable)"
            premium_bias = "BALANCED_SPREADS"
        elif india_vix <= 22.0 or iv_percentile <= 85.0:
            vol_regime = VolatilityRegime.HIGH_IV
            vol_desc = "Elevated Volatility (Options Expensive, Credit Spreads & Iron Condors favored)"
            premium_bias = "SELL_CREDIT_SPREADS"
        else:
            vol_regime = VolatilityRegime.EXTREME_IV
            vol_desc = "Extreme Volatility / Event Risk (High Gamma Danger, Strict Wing Hedging Required)"
            premium_bias = "STRICT_HEDGED_ONLY"

        # 2. Trend Regime
        dist_from_ema20_pct = ((nifty_spot - ema_20) / ema_20) * 100.0
        is_above_50 = nifty_spot > ema_50
        is_above_20 = nifty_spot > ema_20

        if adx_14 > 25.0 and is_above_20 and is_above_50 and dist_from_ema20_pct > 0.5:
            trend = MarketTrend.STRONG_BULLISH
            trend_desc = "Strong Bullish Momentum (Breakout above moving averages with high ADX)"
            directional_bias = "BULLISH"
        elif is_above_20 and is_above_50:
            trend = MarketTrend.BULLISH
            trend_desc = "Moderate Bullish (Trading above EMAs with stable slope)"
            directional_bias = "BULLISH"
        elif adx_14 > 25.0 and not is_above_20 and not is_above_50 and dist_from_ema20_pct < -0.5:
            trend = MarketTrend.STRONG_BEARISH
            trend_desc = "Strong Bearish Breakdown (Trading below moving averages with high ADX)"
            directional_bias = "BEARISH"
        elif not is_above_20 and not is_above_50:
            trend = MarketTrend.BEARISH
            trend_desc = "Moderate Bearish (Trading below EMAs with downward pressure)"
            directional_bias = "BEARISH"
        elif abs(dist_from_ema20_pct) <= 0.4 and adx_14 < 20.0:
            trend = MarketTrend.RANGEBOUND
            trend_desc = "Rangebound Consolidation (Low ADX, Oscillating around EMA-20)"
            directional_bias = "NEUTRAL"
        else:
            trend = MarketTrend.HIGH_VOLATILITY_CHOP
            trend_desc = "Choppy Sideways with Whipsaws (Conflicting indicator signals)"
            directional_bias = "NEUTRAL"

        # 3. PCR & Max Pain Sentiment
        if pcr_oi >= 1.30:
            pcr_sentiment = "Overbought / Strong Put Writing Support"
        elif pcr_oi >= 1.0:
            pcr_sentiment = "Mildly Bullish Put Accumulation"
        elif pcr_oi >= 0.75:
            pcr_sentiment = "Mildly Bearish Call Dominated"
        else:
            pcr_sentiment = "Oversold / Heavy Call Writing Pressure"

        max_pain_diff = nifty_spot - max_pain_strike
        max_pain_gravity = f"Spot is {abs(max_pain_diff):.0f} pts {'above' if max_pain_diff > 0 else 'below'} Max Pain strike ({max_pain_strike:.0f}). Expiry gravitational pull toward {max_pain_strike:.0f}."

        return {
            "nifty_spot": nifty_spot,
            "trend": trend.value,
            "trend_description": trend_desc,
            "directional_bias": directional_bias,
            "volatility_regime": vol_regime.value,
            "volatility_description": vol_desc,
            "premium_bias": premium_bias,
            "india_vix": india_vix,
            "iv_percentile": iv_percentile,
            "pcr_oi": pcr_oi,
            "pcr_sentiment": pcr_sentiment,
            "max_pain_strike": max_pain_strike,
            "max_pain_gravity": max_pain_gravity,
            "adx_14": adx_14,
        }
