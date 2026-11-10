"""
Unit tests for behavioral leakage detection.
"""

import unittest
import os
from trade_analyzer.parser import TradebookParser
from trade_analyzer.metrics import PerformanceMetrics
from trade_analyzer.leakage_detector import LeakageDetector


class TestLeakageDetector(unittest.TestCase):

    def setUp(self):
        sample_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_zerodha_trades.csv")
        _, _, self.trades_df = TradebookParser.parse_csv(sample_path)

    def test_metrics_calculation(self):
        metrics = PerformanceMetrics.calculate_metrics(self.trades_df, initial_capital=200000.0)
        self.assertGreater(metrics["total_trades"], 0)
        self.assertIn("win_rate", metrics)
        self.assertIn("max_drawdown_pct", metrics)
        self.assertIn("sharpe_ratio", metrics)

    def test_leakage_detection(self):
        audit = LeakageDetector.audit_tradebook(self.trades_df, initial_capital=200000.0)
        self.assertIn("behavioral_health_score", audit)
        self.assertIn("total_preventable_loss", audit)
        self.assertGreaterEqual(audit["behavioral_health_score"], 0)
        self.assertLessEqual(audit["behavioral_health_score"], 100)

        # Ensure revenge trading, overtrading, and expiry traps are diagnosed
        self.assertIn("revenge_trading", audit)
        self.assertIn("overtrading", audit)
        self.assertIn("expiry_decay_trap", audit)


if __name__ == "__main__":
    unittest.main()
