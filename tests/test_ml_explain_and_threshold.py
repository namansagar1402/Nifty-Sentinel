import unittest

import numpy as np
import pandas as pd

from backtesting.threshold import threshold_backtest
from ml_engine.classifier import OptionOutcomePredictor
from ml_engine.evaluation import evaluate_models
from ml_engine.explain import explain_forest_prediction
from ml_engine.synthetic_data import generate_synthetic_options_data


class TestExplain(unittest.TestCase):
    def test_contributions_add_up_to_the_models_probability(self):
        predictor = OptionOutcomePredictor()
        predictor.train()
        row = generate_synthetic_options_data(5, seed=3)[predictor.feature_names].iloc[[0]]
        bias, contributions = explain_forest_prediction(predictor.model, row)
        prob = predictor.model.predict_proba(row)[0][1]
        self.assertAlmostEqual(bias + sum(contributions.values()), prob, places=9)
        self.assertEqual(set(contributions), set(predictor.feature_names))


class TestThresholdBacktest(unittest.TestCase):
    def setUp(self):
        self.pnl = np.array([100.0, -50.0, 200.0, -80.0])
        self.prob = np.array([0.9, 0.2, 0.6, 0.55])

    def test_only_trades_above_threshold_are_taken(self):
        res = threshold_backtest(self.pnl, self.prob, threshold=0.58, initial_capital=1000.0)
        self.assertEqual(res["trades"], 2)
        self.assertEqual(res["equity_curve"][-1], 1000.0 + 100.0 + 200.0)
        self.assertEqual(res["win_rate"], 100.0)

    def test_threshold_zero_takes_everything(self):
        res = threshold_backtest(self.pnl, self.prob, threshold=0.0, initial_capital=1000.0)
        self.assertEqual(res["trades"], 4)
        self.assertEqual(len(res["equity_curve"]), 5)
        # Equity 1000 -> 1100 -> 1050 -> 1250 -> 1170: the deepest fall is 1250 -> 1170.
        self.assertAlmostEqual(res["max_drawdown_pct"], (1250 - 1170) / 1250 * 100, places=9)


class TestRuin(unittest.TestCase):
    def test_trading_stops_when_the_account_is_wiped_out(self):
        pnl = np.array([-600.0, -600.0, 5000.0, 5000.0])
        res = threshold_backtest(pnl, np.ones(4), 0.5, initial_capital=1000.0)
        self.assertEqual(res["ruined_after_trades"], 2)
        self.assertEqual(res["trades"], 2)
        self.assertEqual(res["final_capital"], 0.0)
        self.assertEqual(res["equity_curve"], [1000.0, 400.0, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(res["max_drawdown_pct"], 100.0)

    def test_no_ruin_reports_none(self):
        res = threshold_backtest(np.array([10.0, -5.0]), np.ones(2), 0.5, initial_capital=1000.0)
        self.assertIsNone(res["ruined_after_trades"])


class TestThreeModels(unittest.TestCase):
    def test_gradient_boosting_is_compared_too(self):
        rep = evaluate_models(test_samples=500, cv_folds=3)
        self.assertEqual(set(rep["models"]), {"random_forest", "logistic_regression", "gradient_boosting"})
        self.assertGreater(rep["models"]["gradient_boosting"]["roc_auc"], 0.5)


if __name__ == "__main__":
    unittest.main()
