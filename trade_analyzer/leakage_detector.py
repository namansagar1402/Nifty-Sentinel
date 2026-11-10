"""
Behavioral Leakage Diagnostic Engine for Nifty Options Traders.
Detects Revenge Trading, Overtrading, Expiry Decay Traps, Martingale Averaging, and Asymmetric Risk.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from datetime import timedelta
from core.models import ClosedTrade


class LeakageDetector:
    """
    Scans tradebook history and pinpoints specific psychological leakages causing losses.
    """

    @classmethod
    def audit_tradebook(cls, trades_df: pd.DataFrame, initial_capital: float = 200000.0) -> Dict[str, Any]:
        """
        Executes full psychological audit and produces leakage diagnostics with financial quantification.
        """
        if trades_df.empty:
            return cls._empty_audit()

        df = trades_df.copy()
        df["entry_time"] = pd.to_datetime(df["entry_time"])
        df["exit_time"] = pd.to_datetime(df["exit_time"])
        df = df.sort_values(by="entry_time").reset_index(drop=True)

        # 1. Revenge Trading Audit
        revenge_res = cls._audit_revenge_trading(df)

        # 2. Overtrading & Frequency Decay Audit
        overtrading_res = cls._audit_overtrading(df)

        # 3. 0-DTE / Expiry Theta Decay Audit
        expiry_res = cls._audit_expiry_decay(df)

        # 4. Martingale / Loss Averaging Audit
        martingale_res = cls._audit_martingale(df)

        # 5. Asymmetric Risk:Reward Inversion Audit
        rr_res = cls._audit_risk_reward(df)

        # 6. Overall Behavioral Health Score (0 - 100)
        leakage_score, leak_penalty = cls._calculate_behavioral_score(
            revenge_res, overtrading_res, expiry_res, martingale_res, rr_res
        )

        total_preventable_loss = (
            max(0.0, -revenge_res["net_loss"]) +
            max(0.0, -overtrading_res["overtrading_loss"]) +
            max(0.0, -expiry_res["theta_trap_loss"]) +
            max(0.0, -martingale_res["martingale_loss"])
        )

        return {
            "behavioral_health_score": leakage_score,  # 0 to 100 (100 = disciplined master)
            "total_preventable_loss": round(total_preventable_loss, 2),
            "revenge_trading": revenge_res,
            "overtrading": overtrading_res,
            "expiry_decay_trap": expiry_res,
            "martingale_averaging": martingale_res,
            "risk_reward_inversion": rr_res,
            "leakage_breakdown": [
                {
                    "leak_type": "Revenge Trading",
                    "severity": revenge_res["severity"],
                    "financial_damage": abs(revenge_res["net_loss"]),
                    "description": revenge_res["summary"],
                },
                {
                    "leak_type": "Overtrading Spikes",
                    "severity": overtrading_res["severity"],
                    "financial_damage": abs(overtrading_res["overtrading_loss"]),
                    "description": overtrading_res["summary"],
                },
                {
                    "leak_type": "Expiry 0-DTE Theta Trap",
                    "severity": expiry_res["severity"],
                    "financial_damage": abs(expiry_res["theta_trap_loss"]),
                    "description": expiry_res["summary"],
                },
                {
                    "leak_type": "Loss Averaging (Martingale)",
                    "severity": martingale_res["severity"],
                    "financial_damage": abs(martingale_res["martingale_loss"]),
                    "description": martingale_res["summary"],
                },
                {
                    "leak_type": "Risk:Reward Asymmetry",
                    "severity": rr_res["severity"],
                    "financial_damage": abs(rr_res["asymmetry_penalty"]),
                    "description": rr_res["summary"],
                },
            ],
        }

    @staticmethod
    def _audit_revenge_trading(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Flags trades initiated within 15 minutes of closing a losing trade.
        """
        revenge_flags = []
        revenge_pnl = []

        for i in range(len(df)):
            is_revenge = False
            if i > 0:
                prev_trade = df.iloc[i - 1]
                curr_trade = df.iloc[i]

                # If previous trade was a loss
                if prev_trade["net_pnl"] < 0:
                    time_diff = (curr_trade["entry_time"] - prev_trade["exit_time"]).total_seconds() / 60.0
                    # Entered within 15 mins
                    if 0 <= time_diff <= 15.0:
                        is_revenge = True
                        revenge_pnl.append(curr_trade["net_pnl"])

            revenge_flags.append(is_revenge)

        revenge_trades_count = sum(revenge_flags)
        total_trades = len(df)
        net_revenge_loss = sum(revenge_pnl) if revenge_pnl else 0.0

        win_count = sum(1 for p in revenge_pnl if p > 0)
        revenge_win_rate = (win_count / revenge_trades_count * 100.0) if revenge_trades_count > 0 else 0.0

        if revenge_trades_count == 0 or net_revenge_loss >= 0:
            severity = "LOW"
            summary = "No significant revenge trading detected. Good emotional discipline after losing trades."
        elif net_revenge_loss < -15000 or (revenge_trades_count / total_trades) > 0.20:
            severity = "CRITICAL"
            summary = f"Severe Revenge Trading! {revenge_trades_count} trades triggered within 15 mins of a loss, wiping out ₹{abs(net_revenge_loss):,.2f} with a dismal {revenge_win_rate:.1f}% win rate."
        else:
            severity = "MODERATE"
            summary = f"Moderate Revenge Trading: {revenge_trades_count} rapid reentry trades after losses cost ₹{abs(net_revenge_loss):,.2f}."

        return {
            "flagged_count": revenge_trades_count,
            "percentage_of_all_trades": round(revenge_trades_count / total_trades * 100.0, 1) if total_trades > 0 else 0.0,
            "net_loss": round(net_revenge_loss, 2),
            "win_rate": round(revenge_win_rate, 1),
            "severity": severity,
            "summary": summary,
            "recommendation": "Enforce a mandatory 20-minute Cooling-Off Lock after any losing trade.",
        }

    @staticmethod
    def _audit_overtrading(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyzes performance degradation as trade count per day increases.
        """
        df["date"] = pd.to_datetime(df["entry_time"]).dt.date
        daily_counts = df.groupby("date").size().to_dict()

        df["day_trade_seq"] = df.groupby("date").cumcount() + 1

        # Compare Trades 1-3 vs Trades 4+
        early_trades = df[df["day_trade_seq"] <= 3]
        late_trades = df[df["day_trade_seq"] > 3]

        early_pnl = float(early_trades["net_pnl"].sum()) if not early_trades.empty else 0.0
        late_pnl = float(late_trades["net_pnl"].sum()) if not late_trades.empty else 0.0

        early_winrate = (len(early_trades[early_trades["net_pnl"] > 0]) / len(early_trades) * 100.0) if len(early_trades) > 0 else 0.0
        late_winrate = (len(late_trades[late_trades["net_pnl"] > 0]) / len(late_trades) * 100.0) if len(late_trades) > 0 else 0.0

        late_charges = float(late_trades["total_charges"].sum()) if "total_charges" in late_trades.columns else (len(late_trades) * 55.0)

        overtrading_loss = late_pnl if late_pnl < 0 else 0.0

        if len(late_trades) == 0:
            severity = "LOW"
            summary = "Strict trade frequency observed. No overtrading beyond 3 trades/day."
        elif late_pnl < -10000 or (late_winrate < early_winrate - 15.0):
            severity = "HIGH"
            summary = f"Overtrading Deterioration: Trades #4+ in a day produced a net loss of ₹{abs(late_pnl):,.2f} with win rate dropping to {late_winrate:.1f}% (vs {early_winrate:.1f}% for first 3 trades)."
        else:
            severity = "MODERATE"
            summary = f"Mild overtrading: {len(late_trades)} trades executed in high-frequency sessions with ₹{late_charges:,.2f} paid in extra commissions."

        return {
            "early_trades_count": len(early_trades),
            "early_trades_pnl": round(early_pnl, 2),
            "early_win_rate": round(early_winrate, 1),
            "late_trades_count": len(late_trades),
            "late_trades_pnl": round(late_pnl, 2),
            "late_win_rate": round(late_winrate, 1),
            "overtrading_loss": round(overtrading_loss, 2),
            "late_brokerage_wasted": round(late_charges, 2),
            "severity": severity,
            "summary": summary,
            "recommendation": "Cap maximum trades per day to 3-4 executions. Disconnect trading terminal once target or cap is reached.",
        }

    @staticmethod
    def _audit_expiry_decay(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Identifies long option buying trades held on Thursday expiry that suffered >70% loss or expired worthless.
        """
        # Long options on expiry day
        expiry_longs = df[(df["is_expiry_day"] == True) & (df["direction"] == "LONG")]

        decay_trapped = expiry_longs[expiry_longs["pnl_percent"] <= -60.0]
        decay_loss = float(decay_trapped["net_pnl"].sum()) if not decay_trapped.empty else 0.0

        total_expiry_pnl = float(df[df["is_expiry_day"] == True]["net_pnl"].sum())

        if len(decay_trapped) == 0:
            severity = "LOW"
            summary = "Zero 0-DTE theta decay traps. Expiry day options risk managed cleanly."
        elif decay_loss < -20000:
            severity = "CRITICAL"
            summary = f"Expiry 0-DTE Theta Bleed: {len(decay_trapped)} long option positions held until near-zero decay, destroying ₹{abs(decay_loss):,.2f} on expiry days."
        else:
            severity = "MODERATE"
            summary = f"{len(decay_trapped)} option purchases bled into expiry decay, resulting in ₹{abs(decay_loss):,.2f} in losses."

        return {
            "decay_trap_count": len(decay_trapped),
            "theta_trap_loss": round(decay_loss, 2),
            "total_expiry_pnl": round(total_expiry_pnl, 2),
            "severity": severity,
            "summary": summary,
            "recommendation": "Never hold naked Long Options past 1:30 PM on Expiry Day (0-DTE). Switch to Credit Spreads or exit at 30% premium SL.",
        }

    @staticmethod
    def _audit_martingale(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detects loss averaging / Martingale behavior (repeated buy orders on the same symbol on the same day).
        """
        df["date"] = pd.to_datetime(df["entry_time"]).dt.date
        martingale_loss = 0.0
        martingale_events = 0

        grouped = df.groupby(["date", "symbol"])
        for (dt, sym), group in grouped:
            if len(group) > 1:
                # Check if trader entered after a losing trade on the same symbol with same or higher qty
                for i in range(1, len(group)):
                    prev = group.iloc[i - 1]
                    curr = group.iloc[i]
                    if prev["net_pnl"] < 0 and curr["quantity"] >= prev["quantity"]:
                        martingale_events += 1
                        martingale_loss += curr["net_pnl"]

        martingale_loss_val = martingale_loss if martingale_loss < 0 else 0.0

        if martingale_events == 0:
            severity = "LOW"
            summary = "No Martingale averaging found. Position sizing is disciplined."
        elif martingale_loss_val < -10000:
            severity = "HIGH"
            summary = f"Martingale Loss Averaging: {martingale_events} instances of re-buying into losing contracts, generating ₹{abs(martingale_loss_val):,.2f} in aggravated losses."
        else:
            severity = "MODERATE"
            summary = f"Occasional loss averaging ({martingale_events} trades) detected with net impact of ₹{abs(martingale_loss_val):,.2f}."

        return {
            "martingale_events": martingale_events,
            "martingale_loss": round(martingale_loss_val, 2),
            "severity": severity,
            "summary": summary,
            "recommendation": "Strict rule: Never average down a losing options position. Accept the original stop-loss.",
        }

    @staticmethod
    def _audit_risk_reward(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Audits Risk-to-Reward ratio asymmetry (e.g. risking ₹5,000 to make ₹1,000).
        """
        winners = df[df["net_pnl"] > 0]
        losers = df[df["net_pnl"] < 0]

        avg_win = float(winners["net_pnl"].mean()) if len(winners) > 0 else 0.0
        avg_loss = abs(float(losers["net_pnl"].mean())) if len(losers) > 0 else 0.0

        payoff_ratio = (avg_win / avg_loss) if avg_loss > 0 else 1.0
        win_rate = (len(winners) / len(df) * 100.0) if len(df) > 0 else 0.0

        # Required win rate for breakeven = 1 / (1 + PayoffRatio)
        required_win_rate = (1.0 / (1.0 + payoff_ratio) * 100.0) if payoff_ratio > 0 else 50.0

        asymmetry_penalty = 0.0
        if payoff_ratio < 0.7:
            severity = "CRITICAL"
            summary = f"Negative Risk:Reward Asymmetry: Average loss (₹{avg_loss:,.2f}) is {avg_loss/max(1.0, avg_win):.1f}x larger than average win (₹{avg_win:,.2f}). Requires >{required_win_rate:.1f}% win rate just to break even."
            asymmetry_penalty = (avg_loss - avg_win) * len(losers) * 0.5
        elif payoff_ratio < 1.0:
            severity = "MODERATE"
            summary = f"Suboptimal Risk:Reward (1:{payoff_ratio:.2f}). You risk more than you make per trade."
        else:
            severity = "LOW"
            summary = f"Positive Risk:Reward Profile (1:{payoff_ratio:.2f}). Average win (₹{avg_win:,.2f}) exceeds average loss (₹{avg_loss:,.2f})."

        return {
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "payoff_ratio": round(payoff_ratio, 2),
            "actual_win_rate": round(win_rate, 1),
            "required_win_rate_for_breakeven": round(required_win_rate, 1),
            "asymmetry_penalty": round(asymmetry_penalty, 2),
            "severity": severity,
            "summary": summary,
            "recommendation": "Aim for at least 1:1.5 Risk-to-Reward. Cut losers faster and let winners reach defined strike targets.",
        }

    @staticmethod
    def _calculate_behavioral_score(
        revenge: Dict[str, Any],
        overtrade: Dict[str, Any],
        expiry: Dict[str, Any],
        martingale: Dict[str, Any],
        rr: Dict[str, Any],
    ) -> Tuple[int, int]:
        """Calculates 0-100 score where 100 is pristine behavioral discipline."""
        score = 100

        # Penalties based on severity
        penalties = {
            "revenge": 25 if revenge["severity"] == "CRITICAL" else (15 if revenge["severity"] == "HIGH" else (8 if revenge["severity"] == "MODERATE" else 0)),
            "overtrade": 25 if overtrade["severity"] == "HIGH" else (10 if overtrade["severity"] == "MODERATE" else 0),
            "expiry": 20 if expiry["severity"] == "CRITICAL" else (10 if expiry["severity"] == "MODERATE" else 0),
            "martingale": 15 if martingale["severity"] == "HIGH" else (8 if martingale["severity"] == "MODERATE" else 0),
            "rr": 15 if rr["severity"] == "CRITICAL" else (8 if rr["severity"] == "MODERATE" else 0),
        }

        total_penalty = sum(penalties.values())
        final_score = max(5, score - total_penalty)
        return int(final_score), total_penalty

    @staticmethod
    def _empty_audit() -> Dict[str, Any]:
        return {
            "behavioral_health_score": 100,
            "total_preventable_loss": 0.0,
            "revenge_trading": {"flagged_count": 0, "net_loss": 0.0, "severity": "LOW", "summary": "No trades found."},
            "overtrading": {"overtrading_loss": 0.0, "severity": "LOW", "summary": "No trades found."},
            "expiry_decay_trap": {"theta_trap_loss": 0.0, "severity": "LOW", "summary": "No trades found."},
            "martingale_averaging": {"martingale_loss": 0.0, "severity": "LOW", "summary": "No trades found."},
            "risk_reward_inversion": {"payoff_ratio": 1.0, "severity": "LOW", "summary": "No trades found."},
            "leakage_breakdown": [],
        }
