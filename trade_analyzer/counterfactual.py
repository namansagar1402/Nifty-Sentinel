"""
Counterfactual "What-If" Simulation Engine for Nifty Options Trading.
Re-executes historical trades under strict risk containment rules:
1. Daily Loss Hard Stop (e.g. 1.5% or ₹3,000 max daily loss).
2. Max Trades per Day (e.g. max 3 trades/day).
3. Option Premium Stop-Loss Cap (e.g. 30% max loss per trade).
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from core.models import ClosedTrade


class CounterfactualSimulator:
    """
    Simulates the alternative historical PnL curve if strict risk discipline had been enforced.
    """

    @classmethod
    def simulate_containment(
        cls,
        trades_df: pd.DataFrame,
        initial_capital: float = 200000.0,
        max_daily_loss_inr: float = 4000.0,
        max_trades_per_day: int = 3,
        max_option_loss_pct: float = 30.0,
        enable_daily_stop: bool = True,
        enable_trade_limit: bool = True,
        enable_option_sl_cap: bool = True,
    ) -> Dict[str, Any]:
        """
        Runs counterfactual execution and returns side-by-side performance comparisons.
        """
        if trades_df.empty:
            return {"error": "No trades to simulate"}

        df = trades_df.copy()
        df["entry_time"] = pd.to_datetime(df["entry_time"])
        df = df.sort_values(by="entry_time").reset_index(drop=True)
        df["date"] = df["entry_time"].dt.date

        simulated_trades: List[Dict[str, Any]] = []

        # Track simulation state
        unique_dates = df["date"].unique()

        for cur_date in unique_dates:
            day_trades = df[df["date"] == cur_date].copy().reset_index(drop=True)

            daily_running_pnl = 0.0
            daily_trade_count = 0
            day_locked = False

            for idx, trade in day_trades.iterrows():
                # If terminal is locked for the day due to daily loss limit
                if enable_daily_stop and day_locked:
                    continue  # Trade skipped in simulation!

                # If daily trade limit reached
                if enable_trade_limit and daily_trade_count >= max_trades_per_day:
                    continue  # Overtrade skipped in simulation!

                orig_net_pnl = trade["net_pnl"]
                orig_gross_pnl = trade["gross_pnl"]
                charges = trade.get("total_charges", 50.0)
                direction = trade.get("direction", "LONG")
                entry_price = trade["entry_price"]
                exit_price = trade["exit_price"]
                qty = trade["quantity"]

                # Apply Option Stop Loss Cap if it was a catastrophic loss
                sim_gross_pnl = orig_gross_pnl
                sim_net_pnl = orig_net_pnl

                if enable_option_sl_cap and direction == "LONG" and trade["pnl_percent"] < -max_option_loss_pct:
                    # Capped loss
                    capped_exit_price = entry_price * (1.0 - (max_option_loss_pct / 100.0))
                    sim_gross_pnl = (capped_exit_price - entry_price) * qty
                    sim_net_pnl = sim_gross_pnl - charges

                # Check if this trade breaches the daily loss threshold
                if enable_daily_stop and (daily_running_pnl + sim_net_pnl) < -max_daily_loss_inr:
                    # Cap the loss for this trade at the remaining daily budget
                    allowed_loss = -(max_daily_loss_inr + daily_running_pnl)
                    if sim_net_pnl < allowed_loss:
                        sim_net_pnl = allowed_loss
                        sim_gross_pnl = sim_net_pnl + charges
                    day_locked = True

                daily_running_pnl += sim_net_pnl
                daily_trade_count += 1

                simulated_trades.append({
                    "trade_id": trade["trade_id"],
                    "date": cur_date,
                    "entry_time": trade["entry_time"],
                    "symbol": trade["symbol"],
                    "direction": direction,
                    "orig_net_pnl": orig_net_pnl,
                    "sim_net_pnl": round(sim_net_pnl, 2),
                    "charges": charges,
                    "is_simulated": True,
                })

                if enable_daily_stop and daily_running_pnl <= -max_daily_loss_inr:
                    day_locked = True

        sim_df = pd.DataFrame(simulated_trades)

        if sim_df.empty:
            return {"error": "Simulation resulted in zero executed trades"}

        # Calculate actual vs simulated curves
        actual_total_pnl = float(df["net_pnl"].sum())
        sim_total_pnl = float(sim_df["sim_net_pnl"].sum())
        pnl_diff = sim_total_pnl - actual_total_pnl

        # Actual Equity & Drawdown
        df["actual_cum_pnl"] = df["net_pnl"].cumsum()
        df["actual_equity"] = initial_capital + df["actual_cum_pnl"]
        df["actual_peak"] = df["actual_equity"].cummax()
        df["actual_dd"] = df["actual_equity"] - df["actual_peak"]
        df["actual_dd_pct"] = (df["actual_dd"] / df["actual_peak"]) * 100.0

        actual_max_dd_inr = abs(float(df["actual_dd"].min()))
        actual_max_dd_pct = abs(float(df["actual_dd_pct"].min()))

        # Simulated Equity & Drawdown
        sim_df["sim_cum_pnl"] = sim_df["sim_net_pnl"].cumsum()
        sim_df["sim_equity"] = initial_capital + sim_df["sim_cum_pnl"]
        sim_df["sim_peak"] = sim_df["sim_equity"].cummax()
        sim_df["sim_dd"] = sim_df["sim_equity"] - sim_df["sim_peak"]
        sim_df["sim_dd_pct"] = (sim_df["sim_dd"] / sim_df["sim_peak"]) * 100.0

        sim_max_dd_inr = abs(float(sim_df["sim_dd"].min()))
        sim_max_dd_pct = abs(float(sim_df["sim_dd_pct"].min()))

        dd_reduction_pct = ((actual_max_dd_pct - sim_max_dd_pct) / actual_max_dd_pct * 100.0) if actual_max_dd_pct > 0 else 0.0

        # Win Rates
        actual_wins = len(df[df["net_pnl"] > 0])
        actual_win_rate = (actual_wins / len(df) * 100.0) if len(df) > 0 else 0.0

        sim_wins = len(sim_df[sim_df["sim_net_pnl"] > 0])
        sim_win_rate = (sim_wins / len(sim_df) * 100.0) if len(sim_df) > 0 else 0.0

        # Charges Saved
        actual_charges = float(df["total_charges"].sum()) if "total_charges" in df.columns else (len(df) * 50.0)
        sim_charges = float(sim_df["charges"].sum())
        charges_saved = actual_charges - sim_charges

        return {
            "actual_total_pnl": round(actual_total_pnl, 2),
            "simulated_total_pnl": round(sim_total_pnl, 2),
            "net_capital_saved": round(pnl_diff, 2),
            "actual_max_drawdown_pct": round(actual_max_dd_pct, 2),
            "simulated_max_drawdown_pct": round(sim_max_dd_pct, 2),
            "drawdown_reduction_pct": round(dd_reduction_pct, 1),
            "actual_max_drawdown_inr": round(actual_max_dd_inr, 2),
            "simulated_max_drawdown_inr": round(sim_max_dd_inr, 2),
            "actual_trade_count": len(df),
            "simulated_trade_count": len(sim_df),
            "trades_prevented": len(df) - len(sim_df),
            "charges_saved_inr": round(charges_saved, 2),
            "actual_win_rate": round(actual_win_rate, 1),
            "simulated_win_rate": round(sim_win_rate, 1),
            "simulation_dataframe": sim_df,
            "comparison_curve": cls._build_comparison_timeline(df, sim_df, initial_capital),
        }

    @staticmethod
    def _build_comparison_timeline(actual_df: pd.DataFrame, sim_df: pd.DataFrame, initial_cap: float) -> pd.DataFrame:
        """
        Builds a unified chronological timeline for Plotly comparison charts.
        """
        # Aggregate daily
        actual_daily = actual_df.groupby("date")["net_pnl"].sum().reset_index().rename(columns={"net_pnl": "actual_daily_pnl"})
        sim_daily = sim_df.groupby("date")["sim_net_pnl"].sum().reset_index().rename(columns={"sim_net_pnl": "sim_daily_pnl"})

        merged = pd.merge(actual_daily, sim_daily, on="date", how="outer").fillna(0.0).sort_values(by="date")

        merged["actual_cum_pnl"] = merged["actual_daily_pnl"].cumsum()
        merged["sim_cum_pnl"] = merged["sim_daily_pnl"].cumsum()
        merged["actual_equity"] = initial_cap + merged["actual_cum_pnl"]
        merged["sim_equity"] = initial_cap + merged["sim_cum_pnl"]

        # Drawdowns
        merged["actual_peak"] = merged["actual_equity"].cummax()
        merged["actual_dd_pct"] = ((merged["actual_equity"] - merged["actual_peak"]) / merged["actual_peak"]) * 100.0

        merged["sim_peak"] = merged["sim_equity"].cummax()
        merged["sim_dd_pct"] = ((merged["sim_equity"] - merged["sim_peak"]) / merged["sim_peak"]) * 100.0

        return merged
