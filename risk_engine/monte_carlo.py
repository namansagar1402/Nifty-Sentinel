"""
Monte Carlo Simulation & Predictive Drawdown Forecasting Engine.
Runs 1,000+ path bootstrap simulations to predict future drawdown distributions,
worst-case drawdowns at 95% confidence, and Probability of Ruin.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd


class MonteCarloEngine:
    """
    Simulates future equity trajectories and calculates predictive drawdown cones.
    """

    @classmethod
    def run_simulation(
        cls,
        trades_df: pd.DataFrame,
        num_simulations: int = 1000,
        horizon_trades: int = 100,
        initial_capital: float = 200000.0,
        ruin_threshold_pct: float = 30.0,  # Drawdown > 30% considered severe distress
        random_seed: Optional[int] = 42,
    ) -> Dict[str, Any]:
        """
        Runs bootstrap Monte Carlo simulation on past trade returns.
        """
        if trades_df.empty:
            return cls._empty_simulation(initial_capital)

        pnl_pool = trades_df["net_pnl"].values
        if len(pnl_pool) < 5:
            # Not enough trades to sample reliably; fallback to Gaussian estimation
            mean_p = float(np.mean(pnl_pool)) if len(pnl_pool) > 0 else 0.0
            std_p = float(np.std(pnl_pool)) if len(pnl_pool) > 1 else 1000.0
            pnl_pool = np.random.normal(mean_p, max(100.0, std_p), size=20)

        if random_seed is not None:
            np.random.seed(random_seed)

        # Matrix of simulated trades: shape (num_simulations, horizon_trades)
        sampled_pnls = np.random.choice(pnl_pool, size=(num_simulations, horizon_trades), replace=True)

        # Cumulative PnL paths: shape (num_simulations, horizon_trades + 1)
        cum_pnls = np.hstack([np.zeros((num_simulations, 1)), np.cumsum(sampled_pnls, axis=1)])
        equity_paths = initial_capital + cum_pnls

        # Calculate drawdowns across all paths
        # Peak equity up to each step: shape (num_simulations, horizon_trades + 1)
        peak_equity = np.maximum.accumulate(equity_paths, axis=1)
        drawdowns_inr = equity_paths - peak_equity
        drawdowns_pct = (drawdowns_inr / peak_equity) * 100.0  # negative numbers

        # Maximum Drawdown per path
        max_dds_pct = np.abs(np.min(drawdowns_pct, axis=1))  # positive %
        max_dds_inr = np.abs(np.min(drawdowns_inr, axis=1))  # positive ₹

        # Final Equities
        final_equities = equity_paths[:, -1]
        final_pnls = cum_pnls[:, -1]

        # Probability of Ruin (breaching ruin_threshold_pct drawdown)
        ruin_events = np.sum(max_dds_pct >= ruin_threshold_pct)
        prob_of_ruin_pct = (ruin_events / num_simulations) * 100.0

        # Probability of Profit
        profitable_events = np.sum(final_pnls > 0)
        prob_of_profit_pct = (profitable_events / num_simulations) * 100.0

        # Percentile paths for plotting fan chart
        trade_indices = list(range(horizon_trades + 1))
        p5 = np.percentile(equity_paths, 5, axis=0)
        p10 = np.percentile(equity_paths, 10, axis=0)
        p25 = np.percentile(equity_paths, 25, axis=0)
        p50 = np.percentile(equity_paths, 50, axis=0)  # Median
        p75 = np.percentile(equity_paths, 75, axis=0)
        p90 = np.percentile(equity_paths, 90, axis=0)
        p95 = np.percentile(equity_paths, 95, axis=0)

        # Percentile Max Drawdowns
        mdd_median_pct = float(np.percentile(max_dds_pct, 50))
        mdd_95_pct = float(np.percentile(max_dds_pct, 95))  # 95% worst-case max drawdown
        mdd_99_pct = float(np.percentile(max_dds_pct, 99))

        mdd_median_inr = float(np.percentile(max_dds_inr, 50))
        mdd_95_inr = float(np.percentile(max_dds_inr, 95))

        # Sample 15 paths for background trajectory lines in chart
        sample_path_indices = np.random.choice(num_simulations, size=min(15, num_simulations), replace=False)
        sample_paths = [equity_paths[idx].tolist() for idx in sample_path_indices]

        return {
            "num_simulations": num_simulations,
            "horizon_trades": horizon_trades,
            "initial_capital": initial_capital,
            "ruin_threshold_pct": ruin_threshold_pct,
            "prob_of_ruin_pct": round(prob_of_ruin_pct, 1),
            "prob_of_profit_pct": round(prob_of_profit_pct, 1),
            "median_final_equity": round(float(np.median(final_equities)), 2),
            "p5_final_equity": round(float(np.percentile(final_equities, 5)), 2),
            "p95_final_equity": round(float(np.percentile(final_equities, 95)), 2),
            "mdd_median_pct": round(mdd_median_pct, 1),
            "mdd_95_pct": round(mdd_95_pct, 1),
            "mdd_99_pct": round(mdd_99_pct, 1),
            "mdd_median_inr": round(mdd_median_inr, 2),
            "mdd_95_inr": round(mdd_95_inr, 2),
            "trade_indices": trade_indices,
            "percentiles": {
                "p5": [round(float(x), 2) for x in p5],
                "p10": [round(float(x), 2) for x in p10],
                "p25": [round(float(x), 2) for x in p25],
                "p50": [round(float(x), 2) for x in p50],
                "p75": [round(float(x), 2) for x in p75],
                "p90": [round(float(x), 2) for x in p90],
                "p95": [round(float(x), 2) for x in p95],
            },
            "sample_paths": sample_paths,
            "max_drawdowns_pct_distribution": [round(float(x), 1) for x in max_dds_pct],
        }

    @staticmethod
    def _empty_simulation(initial_capital: float) -> Dict[str, Any]:
        return {
            "num_simulations": 0,
            "horizon_trades": 0,
            "initial_capital": initial_capital,
            "ruin_threshold_pct": 30.0,
            "prob_of_ruin_pct": 0.0,
            "prob_of_profit_pct": 0.0,
            "median_final_equity": initial_capital,
            "p5_final_equity": initial_capital,
            "p95_final_equity": initial_capital,
            "mdd_median_pct": 0.0,
            "mdd_95_pct": 0.0,
            "mdd_99_pct": 0.0,
            "mdd_median_inr": 0.0,
            "mdd_95_inr": 0.0,
            "trade_indices": [0],
            "percentiles": {
                "p5": [initial_capital], "p10": [initial_capital], "p25": [initial_capital],
                "p50": [initial_capital], "p75": [initial_capital], "p90": [initial_capital], "p95": [initial_capital],
            },
            "sample_paths": [],
            "max_drawdowns_pct_distribution": [],
        }
