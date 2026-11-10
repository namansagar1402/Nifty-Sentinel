import unittest

import numpy as np
import pandas as pd

from backtesting.engine import BacktestEngine
from ml_engine.synthetic_data import generate_synthetic_options_data
from ml_engine.classifier import get_predictor

FEATURES = ["delta", "gamma", "theta_burn_pct", "vix", "days_to_expiry", "moneyness"]


class TestMLEngine(unittest.TestCase):
    def test_synthetic_data_generation(self):
        df = generate_synthetic_options_data(100)
        self.assertEqual(len(df), 100)
        self.assertIn("delta", df.columns)
        self.assertIn("is_profitable", df.columns)

    def test_same_seed_is_reproducible(self):
        a = generate_synthetic_options_data(50, seed=1)
        b = generate_synthetic_options_data(50, seed=1)
        pd.testing.assert_frame_equal(a, b)

    def test_different_seeds_give_different_data(self):
        a = generate_synthetic_options_data(50, seed=1)
        b = generate_synthetic_options_data(50, seed=2)
        self.assertFalse(a[FEATURES].equals(b[FEATURES]))

    def test_generator_does_not_reset_global_numpy_rng(self):
        np.random.seed(123)
        expected = np.random.random(5)
        np.random.seed(123)
        generate_synthetic_options_data(20)
        actual = np.random.random(5)
        np.testing.assert_array_equal(expected, actual)

    def test_backtest_data_does_not_overlap_training_data(self):
        predictor = get_predictor()
        engine = BacktestEngine(n_trades=200)
        self.assertNotEqual(engine.data_seed, predictor.training_seed)

        train = generate_synthetic_options_data(5000, seed=predictor.training_seed)
        test = generate_synthetic_options_data(200, seed=engine.data_seed)
        overlap = test[FEATURES].merge(train[FEATURES], how="inner")
        self.assertEqual(len(overlap), 0)

    def test_predictor_initialization_and_prediction(self):
        predictor = get_predictor()
        self.assertTrue(predictor.is_trained)

        result = predictor.predict_trade(0.5, 0.05, 5.0, 15.0, 10, 1.0)
        self.assertIn(result, [0, 1])

        df = generate_synthetic_options_data(10)
        results = predictor.predict_batch(df)
        self.assertEqual(len(results), 10)


if __name__ == "__main__":
    unittest.main()
