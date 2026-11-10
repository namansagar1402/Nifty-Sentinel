"""
Unit tests for Monte Carlo, VaR/CVaR, and Counterfactual simulation.
"""

import unittest
import os
from trade_analyzer.parser import TradebookParser
from trade_analyzer.counterfactual import CounterfactualSimulator
from risk_engine.var_cvar import RiskValueEngine
from risk_engine.monte_carlo import MonteCarloEngine
from risk_engine.containment import DrawdownContainmentEngine
from strategy_optimizer.strategy_recommender import StrategyRecommender


class TestRiskAndCounterfactual(unittest.TestCase):

    def setUp(self):
        sample_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_zerodha_trades.csv")
        _, _, self.trades_df = TradebookParser.parse_csv(sample_path)

    def test_var_cvar_calculation(self):
        res = RiskValueEngine.calculate_var_cvar(self.trades_df, current_equity=200000.0)
        self.assertIn("var_metrics", res)
        self.assertIn("95%", res["var_metrics"])
        self.assertIn("99%", res["var_metrics"])

    def test_monte_carlo_simulation(self):
        mc = MonteCarloEngine.run_simulation(self.trades_df, num_simulations=100, horizon_trades=50, initial_capital=200000.0)
        self.assertEqual(mc["num_simulations"], 100)
        self.assertIn("prob_of_ruin_pct", mc)
        self.assertIn("mdd_95_pct", mc)
        self.assertIn("percentiles", mc)

    def test_counterfactual_simulation(self):
        sim = CounterfactualSimulator.simulate_containment(
            self.trades_df,
            initial_capital=200000.0,
            max_daily_loss_inr=4000.0,
            max_trades_per_day=3,
            max_option_loss_pct=30.0,
        )
        self.assertIn("actual_total_pnl", sim)
        self.assertIn("simulated_total_pnl", sim)
        self.assertIn("net_capital_saved", sim)
        self.assertIn("drawdown_reduction_pct", sim)

    def test_circuit_breaker_and_sizing(self):
        cb_green = DrawdownContainmentEngine.evaluate_circuit_breaker(
            current_daily_pnl=-1000.0, daily_loss_limit_inr=4000.0, consecutive_losses=0, daily_trade_count=1
        )
        self.assertTrue(cb_green["can_trade"])

        cb_red = DrawdownContainmentEngine.evaluate_circuit_breaker(
            current_daily_pnl=-4500.0, daily_loss_limit_inr=4000.0, consecutive_losses=3, daily_trade_count=4
        )
        self.assertFalse(cb_red["can_trade"])

        sizing = DrawdownContainmentEngine.calculate_position_sizing(
            account_capital=200000.0, win_rate_pct=55.0, payoff_ratio=1.5
        )
        self.assertGreater(sizing["recommended_lots"], 0)

    def test_strategy_recommendation(self):
        recs = StrategyRecommender.get_recommendations(nifty_spot=24500.0, india_vix=14.5)
        self.assertIn("regime_summary", recs)
        self.assertIn("recommended_strategies", recs)
        self.assertGreater(len(recs["recommended_strategies"]), 0)


if __name__ == "__main__":
    unittest.main()
