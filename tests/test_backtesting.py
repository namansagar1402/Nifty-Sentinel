import unittest
from backtesting.engine import BacktestEngine


class TestBacktestingEngine(unittest.TestCase):
    def test_run_backtest(self):
        engine = BacktestEngine(n_trades=100)  # Small number for quick test
        results = engine.run_backtest()

        self.assertIn("baseline", results)
        self.assertIn("rule_based", results)
        self.assertIn("ml_based", results)
        self.assertIn("feature_importances", results)

        for key in ("baseline", "rule_based", "ml_based"):
            self.assertIn("max_drawdown", results[key])
            self.assertIn("success_rate", results[key])
            self.assertGreaterEqual(results[key]["success_rate"], 0)
            self.assertLessEqual(results[key]["success_rate"], 100)

        # Check that equity curves have expected lengths
        self.assertEqual(len(results["baseline"]["equity_curve"]), 101)  # 100 trades + initial


if __name__ == "__main__":
    unittest.main()
