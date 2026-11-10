"""
Risk Command Center: gathers the regime classifier, strategy recommender, VaR/CVaR,
Monte Carlo, circuit breaker and Kelly sizer into one snapshot for the dashboard.

Every input is computed here (no UI code), so the whole snapshot is unit-testable.
"""

from datetime import date
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from core.models import CircuitBreakerStatus
from risk_engine.containment import DrawdownContainmentEngine
from risk_engine.monte_carlo import MonteCarloEngine
from risk_engine.var_cvar import RiskValueEngine
from strategy_optimizer.regime_classifier import MarketRegimeClassifier
from strategy_optimizer.strategy_recommender import StrategyRecommender
from trade_analyzer.metrics import PerformanceMetrics

ADX_PERIOD = 14
DEFAULT_IV_PERCENTILE = 50.0
DEFAULT_STOP_LOSS_POINTS = 30.0
MONTE_CARLO_PATHS = 1000
MONTE_CARLO_HORIZON = 100

_BREAKER_LIGHTS = {
    CircuitBreakerStatus.NORMAL_GREEN.value: "green",
    CircuitBreakerStatus.WARNING_AMBER.value: "amber",
    CircuitBreakerStatus.LOCKED_RED.value: "red",
}


def compute_pcr(chain: pd.DataFrame) -> float:
    """Put-call ratio by open interest."""
    call_oi = float(chain["ce_oi"].sum())
    return float(chain["pe_oi"].sum()) / call_oi if call_oi > 0 else 1.0


def compute_max_pain(chain: pd.DataFrame) -> float:
    """Strike at which option holders would collect the least at expiry."""
    strikes = chain["strike"].to_numpy(dtype=float)
    ce_oi = chain["ce_oi"].to_numpy(dtype=float)
    pe_oi = chain["pe_oi"].to_numpy(dtype=float)
    payouts = [
        float(np.sum(np.maximum(0.0, s - strikes) * ce_oi) + np.sum(np.maximum(0.0, strikes - s) * pe_oi))
        for s in strikes
    ]
    return float(strikes[int(np.argmin(payouts))])


def compute_adx(candles: pd.DataFrame, period: int = ADX_PERIOD) -> float:
    """Wilder's Average Directional Index (0-100); above 25 means a strong trend."""
    if len(candles) < period * 2:
        return 20.0
    high, low, close = candles["High"], candles["Low"], candles["Close"]
    up, down = high.diff(), -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    true_range = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1
    ).max(axis=1)

    alpha = 1.0 / period
    atr = true_range.ewm(alpha=alpha, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=candles.index).ewm(alpha=alpha, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=candles.index).ewm(alpha=alpha, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=alpha, adjust=False).mean().iloc[-1]
    return float(np.clip(adx, 0.0, 100.0)) if pd.notna(adx) else 20.0


def compute_iv_percentile(current_vix: float, vix_history: pd.Series) -> Optional[float]:
    """Share of past VIX readings below today's (0-100). None when there is no history."""
    history = vix_history.dropna()
    if history.empty:
        return None
    return round(float((history < current_vix).mean() * 100.0), 1)


def latest_day_breaker_inputs(trades_df: pd.DataFrame) -> Tuple[float, int, int, Optional[date]]:
    """(day P&L, losing streak at end of day, trade count, date) for the most recent trading day."""
    if trades_df.empty:
        return 0.0, 0, 0, None
    df = trades_df.sort_values("entry_time")
    days = pd.to_datetime(df["entry_time"]).dt.date
    last_day = days.iloc[-1]
    day_trades = df[days == last_day]

    streak = 0
    for pnl in reversed(day_trades["net_pnl"].tolist()):
        if pnl >= 0:
            break
        streak += 1
    return float(day_trades["net_pnl"].sum()), streak, len(day_trades), last_day


def build_risk_snapshot(
    spot: float,
    vix: float,
    candles: pd.DataFrame,
    chain: pd.DataFrame,
    trades_df: pd.DataFrame,
    capital: float = 200000.0,
    vix_history: Optional[pd.Series] = None,
    days_to_expiry: int = 5,
    stop_loss_points: float = DEFAULT_STOP_LOSS_POINTS,
) -> Dict[str, Any]:
    """Runs every risk engine once and returns everything the Command Center tab shows."""
    ema_20 = float(candles["EMA20"].iloc[-1]) if "EMA20" in candles and not candles.empty else spot
    ema_50 = float(candles["EMA50"].iloc[-1]) if "EMA50" in candles and not candles.empty else spot
    iv_pct = compute_iv_percentile(vix, vix_history if vix_history is not None else pd.Series(dtype=float))
    inputs = {
        "ema_20": round(ema_20, 2),
        "ema_50": round(ema_50, 2),
        "adx_14": round(compute_adx(candles), 1),
        "pcr_oi": round(compute_pcr(chain), 2),
        "max_pain": compute_max_pain(chain),
        "iv_percentile": iv_pct if iv_pct is not None else DEFAULT_IV_PERCENTILE,
        "iv_percentile_estimated": iv_pct is None,
    }

    recs = StrategyRecommender.get_recommendations(
        nifty_spot=spot, india_vix=vix, iv_percentile=inputs["iv_percentile"],
        ema_20=ema_20, ema_50=ema_50, pcr_oi=inputs["pcr_oi"], max_pain=inputs["max_pain"],
        days_to_expiry=days_to_expiry,
    )
    # The recommender doesn't take ADX; re-classify with it so the regime card is complete.
    regime = MarketRegimeClassifier.classify_regime(
        nifty_spot=spot, ema_20=ema_20, ema_50=ema_50, india_vix=vix,
        iv_percentile=inputs["iv_percentile"], pcr_oi=inputs["pcr_oi"],
        max_pain_strike=inputs["max_pain"], adx_14=inputs["adx_14"],
    )

    day_pnl, streak, day_count, day = latest_day_breaker_inputs(trades_df)
    daily_limit = DrawdownContainmentEngine.calculate_adaptive_daily_limit(capital, india_vix=vix)
    breaker = DrawdownContainmentEngine.evaluate_circuit_breaker(
        current_daily_pnl=day_pnl, daily_loss_limit_inr=daily_limit,
        consecutive_losses=streak, daily_trade_count=day_count,
    )
    breaker["light"] = _BREAKER_LIGHTS.get(breaker["status"], "amber")
    breaker["replay_date"] = day

    metrics = PerformanceMetrics.calculate_metrics(trades_df, initial_capital=capital)
    sizing = DrawdownContainmentEngine.calculate_position_sizing(
        account_capital=capital, win_rate_pct=metrics.get("win_rate", 0.0),
        payoff_ratio=metrics.get("payoff_ratio", 0.0), nifty_spot=spot, sl_points=stop_loss_points,
    )

    return {
        "inputs": inputs,
        "regime": regime,
        "strategies": recs["recommended_strategies"],
        "top_strategy": recs["top_strategy"],
        "var": RiskValueEngine.calculate_var_cvar(trades_df, current_equity=capital),
        "monte_carlo": MonteCarloEngine.run_simulation(
            trades_df, num_simulations=MONTE_CARLO_PATHS, horizon_trades=MONTE_CARLO_HORIZON,
            initial_capital=capital,
        ),
        "breaker": breaker,
        "sizing": sizing,
    }
