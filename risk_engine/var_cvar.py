"""
Value-at-Risk (VaR) and Conditional Value-at-Risk (CVaR / Expected Shortfall) Engine.
Calculates Parametric VaR, Historical VaR, and tail risk metrics for Nifty F&O options trading.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import norm, t


class RiskValueEngine:
    """
    Computes statistical tail risk, VaR, and Expected Shortfall on daily and per-trade returns.
    """

    @classmethod
    def calculate_var_cvar(
        cls,
        trades_df: pd.DataFrame,
        confidence_levels: Tuple[float, float] = (0.95, 0.99),
        current_equity: float = 200000.0,
    ) -> Dict[str, Any]:
        """
        Calculates Historical and Parametric VaR / CVaR for the trader's historical record.
        """
        if trades_df.empty:
            return cls._empty_var_cvar(current_equity)

        df = trades_df.copy()
        pnl_series = df["net_pnl"].values

        # Per-Trade Metrics
        trade_count = len(pnl_series)
        mean_pnl = float(np.mean(pnl_series))
        std_pnl = float(np.std(pnl_series, ddof=1)) if trade_count > 1 else 0.0

        # Daily Returns
        df["date"] = pd.to_datetime(df["entry_time"]).dt.date
        daily_pnl = df.groupby("date")["net_pnl"].sum().values
        mean_daily = float(np.mean(daily_pnl)) if len(daily_pnl) > 0 else 0.0
        std_daily = float(np.std(daily_pnl, ddof=1)) if len(daily_pnl) > 1 else 0.0

        # 95% and 99% VaR & CVaR (Daily)
        var_results = {}
        for conf in confidence_levels:
            alpha = 1.0 - conf
            z_score = norm.ppf(conf)

            # Parametric Daily VaR (in ₹)
            param_var_daily_inr = -(mean_daily - z_score * std_daily)
            param_var_daily_pct = (param_var_daily_inr / current_equity) * 100.0

            # Historical Daily VaR (Percentile)
            if len(daily_pnl) >= 5:
                hist_cutoff = float(np.percentile(daily_pnl, alpha * 100.0))
                hist_var_daily_inr = -hist_cutoff if hist_cutoff < 0 else 0.0
                # CVaR: Mean of losses exceeding VaR
                tail_losses = daily_pnl[daily_pnl <= hist_cutoff]
                cvar_daily_inr = -float(np.mean(tail_losses)) if len(tail_losses) > 0 else hist_var_daily_inr
            else:
                hist_var_daily_inr = param_var_daily_inr
                cvar_daily_inr = param_var_daily_inr * 1.25

            hist_var_daily_pct = (hist_var_daily_inr / current_equity) * 100.0
            cvar_daily_pct = (cvar_daily_inr / current_equity) * 100.0

            # Parametric Per-Trade VaR
            param_var_trade_inr = -(mean_pnl - z_score * std_pnl)
            
            # Historical Per-Trade VaR & CVaR
            hist_trade_cutoff = float(np.percentile(pnl_series, alpha * 100.0))
            hist_var_trade_inr = -hist_trade_cutoff if hist_trade_cutoff < 0 else 0.0
            tail_trade_losses = pnl_series[pnl_series <= hist_trade_cutoff]
            cvar_trade_inr = -float(np.mean(tail_trade_losses)) if len(tail_trade_losses) > 0 else hist_var_trade_inr

            conf_label = f"{int(conf * 100)}%"
            var_results[conf_label] = {
                "confidence": conf,
                "daily_parametric_var_inr": round(max(0.0, param_var_daily_inr), 2),
                "daily_parametric_var_pct": round(max(0.0, param_var_daily_pct), 2),
                "daily_historical_var_inr": round(max(0.0, hist_var_daily_inr), 2),
                "daily_historical_var_pct": round(max(0.0, hist_var_daily_pct), 2),
                "daily_cvar_expected_shortfall_inr": round(max(0.0, cvar_daily_inr), 2),
                "daily_cvar_expected_shortfall_pct": round(max(0.0, cvar_daily_pct), 2),
                "trade_historical_var_inr": round(max(0.0, hist_var_trade_inr), 2),
                "trade_cvar_expected_shortfall_inr": round(max(0.0, cvar_trade_inr), 2),
            }

        # Tail Risk Assessment
        var_95_daily = var_results["95%"]["daily_historical_var_inr"]
        cvar_95_daily = var_results["95%"]["daily_cvar_expected_shortfall_inr"]

        tail_risk_ratio = (cvar_95_daily / var_95_daily) if var_95_daily > 0 else 1.0
        is_fat_tailed = tail_risk_ratio > 1.4

        return {
            "current_equity": current_equity,
            "daily_mean_pnl": round(mean_daily, 2),
            "daily_std_dev": round(std_daily, 2),
            "trade_mean_pnl": round(mean_pnl, 2),
            "trade_std_dev": round(std_pnl, 2),
            "is_fat_tailed": is_fat_tailed,
            "tail_risk_ratio": round(tail_risk_ratio, 2),
            "var_metrics": var_results,
            "daily_loss_distribution": [float(x) for x in daily_pnl],
        }

    @staticmethod
    def _empty_var_cvar(current_equity: float) -> Dict[str, Any]:
        return {
            "current_equity": current_equity,
            "daily_mean_pnl": 0.0,
            "daily_std_dev": 0.0,
            "trade_mean_pnl": 0.0,
            "trade_std_dev": 0.0,
            "is_fat_tailed": False,
            "tail_risk_ratio": 1.0,
            "var_metrics": {
                "95%": {
                    "confidence": 0.95,
                    "daily_parametric_var_inr": 0.0,
                    "daily_parametric_var_pct": 0.0,
                    "daily_historical_var_inr": 0.0,
                    "daily_historical_var_pct": 0.0,
                    "daily_cvar_expected_shortfall_inr": 0.0,
                    "daily_cvar_expected_shortfall_pct": 0.0,
                    "trade_historical_var_inr": 0.0,
                    "trade_cvar_expected_shortfall_inr": 0.0,
                },
                "99%": {
                    "confidence": 0.99,
                    "daily_parametric_var_inr": 0.0,
                    "daily_parametric_var_pct": 0.0,
                    "daily_historical_var_inr": 0.0,
                    "daily_historical_var_pct": 0.0,
                    "daily_cvar_expected_shortfall_inr": 0.0,
                    "daily_cvar_expected_shortfall_pct": 0.0,
                    "trade_historical_var_inr": 0.0,
                    "trade_cvar_expected_shortfall_inr": 0.0,
                },
            },
            "daily_loss_distribution": [],
        }
