"""
Quantitative performance metrics and statistical analytics engine for options trading.
Calculates Drawdowns, Expectancy, Profit Factor, Sharpe/Sortino, and Segment Breakdowns.
"""

import math
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from core.models import ClosedTrade


class PerformanceMetrics:
    """
    Computes professional quantitative metrics from closed trade records.
    """

    @staticmethod
    def calculate_metrics(trades_df: pd.DataFrame, initial_capital: float = 200000.0) -> Dict[str, Any]:
        """
        Calculates all key statistics from closed trades DataFrame.
        """
        if trades_df.empty:
            return PerformanceMetrics._empty_metrics(initial_capital)

        df = trades_df.copy()
        if "net_pnl" not in df.columns:
            df["net_pnl"] = df["gross_pnl"] - df.get("total_charges", 0.0)

        # Sort chronologically
        df["entry_time"] = pd.to_datetime(df["entry_time"])
        df = df.sort_values(by="entry_time").reset_index(drop=True)

        total_trades = len(df)
        winning_trades = df[df["net_pnl"] > 0]
        losing_trades = df[df["net_pnl"] < 0]
        breakeven_trades = df[df["net_pnl"] == 0]

        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        win_rate = (win_count / total_trades * 100.0) if total_trades > 0 else 0.0
        loss_rate = (loss_count / total_trades * 100.0) if total_trades > 0 else 0.0

        gross_pnl = float(df["gross_pnl"].sum())
        total_charges = float(df["total_charges"].sum()) if "total_charges" in df.columns else 0.0
        net_pnl = float(df["net_pnl"].sum())

        total_profit = float(winning_trades["net_pnl"].sum()) if win_count > 0 else 0.0
        total_loss = abs(float(losing_trades["net_pnl"].sum())) if loss_count > 0 else 0.0

        profit_factor = (total_profit / total_loss) if total_loss > 0 else (99.0 if total_profit > 0 else 0.0)

        avg_win = (total_profit / win_count) if win_count > 0 else 0.0
        avg_loss = (total_loss / loss_count) if loss_count > 0 else 0.0
        payoff_ratio = (avg_win / avg_loss) if avg_loss > 0 else (avg_win if avg_win > 0 else 0.0)

        # Expectancy = (Win% * AvgWin) - (Loss% * AvgLoss) - AvgChargesPerTrade
        avg_charges = total_charges / total_trades if total_trades > 0 else 0.0
        expectancy_inr = ((win_rate / 100.0) * avg_win) - ((loss_rate / 100.0) * avg_loss)
        expectancy_ratio = (expectancy_inr / avg_loss) if avg_loss > 0 else 0.0

        # Equity Curve and Drawdown
        df["cum_net_pnl"] = df["net_pnl"].cumsum()
        df["equity"] = initial_capital + df["cum_net_pnl"]
        df["peak_equity"] = df["equity"].cummax()
        df["drawdown_inr"] = df["equity"] - df["peak_equity"]
        df["drawdown_pct"] = (df["drawdown_inr"] / df["peak_equity"]) * 100.0

        max_drawdown_inr = abs(float(df["drawdown_inr"].min()))
        max_drawdown_pct = abs(float(df["drawdown_pct"].min()))
        current_drawdown_inr = abs(float(df["drawdown_inr"].iloc[-1]))
        current_drawdown_pct = abs(float(df["drawdown_pct"].iloc[-1]))

        # Daily Returns & Sharpe/Sortino
        df["date"] = pd.to_datetime(df["entry_time"]).dt.date
        daily_df = df.groupby("date").agg(
            daily_net_pnl=("net_pnl", "sum"),
            daily_trades=("trade_id", "count"),
            daily_gross=("gross_pnl", "sum"),
            daily_charges=("total_charges", "sum"),
        ).reset_index()

        daily_df["cum_pnl"] = daily_df["daily_net_pnl"].cumsum()
        daily_df["equity"] = initial_capital + daily_df["cum_pnl"]
        daily_df["daily_return"] = daily_df["daily_net_pnl"] / (daily_df["equity"] - daily_df["daily_net_pnl"])

        daily_returns = daily_df["daily_return"].fillna(0.0)
        mean_ret = float(daily_returns.mean()) if len(daily_returns) > 0 else 0.0
        std_ret = float(daily_returns.std()) if len(daily_returns) > 1 else 0.0

        risk_free_daily = 0.07 / 252.0  # 7% annual risk-free rate
        excess_daily = mean_ret - risk_free_daily
        sharpe_ratio = (excess_daily / std_ret * math.sqrt(252)) if std_ret > 1e-6 else 0.0

        downside_returns = daily_returns[daily_returns < 0]
        downside_std = float(downside_returns.std()) if len(downside_returns) > 1 else 0.0
        sortino_ratio = (excess_daily / downside_std * math.sqrt(252)) if downside_std > 1e-6 else 0.0

        total_trading_days = len(daily_df)
        annualized_return = (net_pnl / initial_capital) * (252 / total_trading_days) if total_trading_days > 0 else 0.0
        calmar_ratio = (annualized_return / (max_drawdown_pct / 100.0)) if max_drawdown_pct > 0 else 0.0

        # Winning and Losing Streaks
        pnl_signs = (df["net_pnl"] > 0).astype(int)
        max_win_streak = 0
        max_loss_streak = 0
        cur_win_streak = 0
        cur_loss_streak = 0

        for sign in pnl_signs:
            if sign == 1:
                cur_win_streak += 1
                cur_loss_streak = 0
                max_win_streak = max(max_win_streak, cur_win_streak)
            else:
                cur_loss_streak += 1
                cur_win_streak = 0
                max_loss_streak = max(max_loss_streak, cur_loss_streak)

        # Segment Breakdowns: CE vs PE
        ce_trades = df[df["option_type"] == "CE"]
        pe_trades = df[df["option_type"] == "PE"]
        ce_pnl = float(ce_trades["net_pnl"].sum()) if not ce_trades.empty else 0.0
        pe_pnl = float(pe_trades["net_pnl"].sum()) if not pe_trades.empty else 0.0
        ce_win_rate = (len(ce_trades[ce_trades["net_pnl"] > 0]) / len(ce_trades) * 100.0) if len(ce_trades) > 0 else 0.0
        pe_win_rate = (len(pe_trades[pe_trades["net_pnl"] > 0]) / len(pe_trades) * 100.0) if len(pe_trades) > 0 else 0.0

        # Segment Breakdowns: Option Buying (LONG) vs Option Selling (SHORT)
        long_trades = df[df["direction"] == "LONG"]
        short_trades = df[df["direction"] == "SHORT"]
        long_pnl = float(long_trades["net_pnl"].sum()) if not long_trades.empty else 0.0
        short_pnl = float(short_trades["net_pnl"].sum()) if not short_trades.empty else 0.0
        long_win_rate = (len(long_trades[long_trades["net_pnl"] > 0]) / len(long_trades) * 100.0) if len(long_trades) > 0 else 0.0
        short_win_rate = (len(short_trades[short_trades["net_pnl"] > 0]) / len(short_trades) * 100.0) if len(short_trades) > 0 else 0.0

        # Expiry Day Breakdown
        expiry_trades = df[df["is_expiry_day"] == True]
        non_expiry_trades = df[df["is_expiry_day"] == False]
        expiry_pnl = float(expiry_trades["net_pnl"].sum()) if not expiry_trades.empty else 0.0
        non_expiry_pnl = float(non_expiry_trades["net_pnl"].sum()) if not non_expiry_trades.empty else 0.0

        # Best / Worst Day
        best_day_pnl = float(daily_df["daily_net_pnl"].max()) if not daily_df.empty else 0.0
        worst_day_pnl = float(daily_df["daily_net_pnl"].min()) if not daily_df.empty else 0.0

        # Max Drawdown Duration (Trades and Days)
        drawdown_duration_days = PerformanceMetrics._calculate_max_drawdown_duration(daily_df)

        return {
            "initial_capital": initial_capital,
            "final_equity": initial_capital + net_pnl,
            "net_pnl": round(net_pnl, 2),
            "gross_pnl": round(gross_pnl, 2),
            "total_charges": round(total_charges, 2),
            "total_trades": total_trades,
            "winning_trades": win_count,
            "losing_trades": loss_count,
            "breakeven_trades": len(breakeven_trades),
            "win_rate": round(win_rate, 2),
            "loss_rate": round(loss_rate, 2),
            "profit_factor": round(profit_factor, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "payoff_ratio": round(payoff_ratio, 2),
            "expectancy_inr": round(expectancy_inr, 2),
            "expectancy_ratio": round(expectancy_ratio, 2),
            "max_drawdown_inr": round(max_drawdown_inr, 2),
            "max_drawdown_pct": round(max_drawdown_pct, 2),
            "current_drawdown_inr": round(current_drawdown_inr, 2),
            "current_drawdown_pct": round(current_drawdown_pct, 2),
            "max_drawdown_duration_days": drawdown_duration_days,
            "sharpe_ratio": round(sharpe_ratio, 2),
            "sortino_ratio": round(sortino_ratio, 2),
            "calmar_ratio": round(calmar_ratio, 2),
            "max_win_streak": max_win_streak,
            "max_loss_streak": max_loss_streak,
            "best_day_pnl": round(best_day_pnl, 2),
            "worst_day_pnl": round(worst_day_pnl, 2),
            "total_trading_days": total_trading_days,
            "avg_trades_per_day": round(total_trades / total_trading_days, 1) if total_trading_days > 0 else 0.0,
            "ce_breakdown": {
                "count": len(ce_trades),
                "pnl": round(ce_pnl, 2),
                "win_rate": round(ce_win_rate, 2),
            },
            "pe_breakdown": {
                "count": len(pe_trades),
                "pnl": round(pe_pnl, 2),
                "win_rate": round(pe_win_rate, 2),
            },
            "long_buyer_breakdown": {
                "count": len(long_trades),
                "pnl": round(long_pnl, 2),
                "win_rate": round(long_win_rate, 2),
            },
            "short_seller_breakdown": {
                "count": len(short_trades),
                "pnl": round(short_pnl, 2),
                "win_rate": round(short_win_rate, 2),
            },
            "expiry_breakdown": {
                "expiry_day_trades": len(expiry_trades),
                "expiry_day_pnl": round(expiry_pnl, 2),
                "non_expiry_day_pnl": round(non_expiry_pnl, 2),
            },
            "daily_df": daily_df,
            "equity_curve": df[["entry_time", "equity", "drawdown_inr", "drawdown_pct", "net_pnl", "cum_net_pnl"]],
        }

    @staticmethod
    def _calculate_max_drawdown_duration(daily_df: pd.DataFrame) -> int:
        if daily_df.empty or len(daily_df) < 2:
            return 0
        
        peak = daily_df["equity"].iloc[0]
        max_duration = 0
        current_duration = 0

        for eq in daily_df["equity"]:
            if eq >= peak:
                peak = eq
                current_duration = 0
            else:
                current_duration += 1
                max_duration = max(max_duration, current_duration)

        return max_duration

    @staticmethod
    def _empty_metrics(initial_capital: float) -> Dict[str, Any]:
        return {
            "initial_capital": initial_capital,
            "final_equity": initial_capital,
            "net_pnl": 0.0,
            "gross_pnl": 0.0,
            "total_charges": 0.0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "breakeven_trades": 0,
            "win_rate": 0.0,
            "loss_rate": 0.0,
            "profit_factor": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "payoff_ratio": 0.0,
            "expectancy_inr": 0.0,
            "expectancy_ratio": 0.0,
            "max_drawdown_inr": 0.0,
            "max_drawdown_pct": 0.0,
            "current_drawdown_inr": 0.0,
            "current_drawdown_pct": 0.0,
            "max_drawdown_duration_days": 0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "max_win_streak": 0,
            "max_loss_streak": 0,
            "best_day_pnl": 0.0,
            "worst_day_pnl": 0.0,
            "total_trading_days": 0,
            "avg_trades_per_day": 0.0,
            "ce_breakdown": {"count": 0, "pnl": 0.0, "win_rate": 0.0},
            "pe_breakdown": {"count": 0, "pnl": 0.0, "win_rate": 0.0},
            "long_buyer_breakdown": {"count": 0, "pnl": 0.0, "win_rate": 0.0},
            "short_seller_breakdown": {"count": 0, "pnl": 0.0, "win_rate": 0.0},
            "expiry_breakdown": {"expiry_day_trades": 0, "expiry_day_pnl": 0.0, "non_expiry_day_pnl": 0.0},
            "daily_df": pd.DataFrame(),
            "equity_curve": pd.DataFrame(),
        }
