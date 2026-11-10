import unittest

import numpy as np
import pandas as pd

from ml_engine.real_data import build_option_trades, time_split

FEATURES = ["delta", "gamma", "theta_burn_pct", "vix", "days_to_expiry", "moneyness"]


def _history(closes, vix=15.0, start="2025-01-01"):
    dates = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame({"date": dates, "close": np.asarray(closes, dtype=float), "vix": vix})


class TestBuildOptionTrades(unittest.TestCase):
    def setUp(self):
        # Steady 0.5%-a-day rise for 40 business days.
        self.hist = _history(25000 * 1.005 ** np.arange(40))
        self.trades = build_option_trades(self.hist, dte_choices=(7,), offsets_pct=(0.0,))

    def test_has_model_features_and_labels(self):
        for col in FEATURES + ["is_profitable", "pnl_inr", "entry_date", "exit_date", "option_type"]:
            self.assertIn(col, self.trades.columns)
        self.assertTrue(self.trades["is_profitable"].isin([0, 1]).all())

    def test_exit_is_last_trading_day_on_or_before_expiry(self):
        row = self.trades.iloc[0]
        self.assertGreater(row["exit_date"], row["entry_date"])
        self.assertLessEqual((row["exit_date"] - row["entry_date"]).days, 7)
        later = self.hist[(self.hist["date"] > row["exit_date"])
                          & (self.hist["date"] <= row["entry_date"] + pd.Timedelta(days=7))]
        self.assertTrue(later.empty)

    def test_label_is_intrinsic_value_against_premium(self):
        row = self.trades[self.trades["option_type"] == "CE"].iloc[0]
        exit_close = self.hist.loc[self.hist["date"] == row["exit_date"], "close"].iloc[0]
        intrinsic = max(0.0, exit_close - row["strike"])
        self.assertAlmostEqual(row["pnl_inr"], (intrinsic - row["premium"]) * 50, places=4)
        self.assertEqual(row["is_profitable"], int(row["pnl_inr"] > 0))

    def test_rising_market_rewards_calls_over_puts(self):
        by_type = self.trades.groupby("option_type")["is_profitable"].mean()
        self.assertGreater(by_type["CE"], by_type["PE"])

    def test_no_trade_runs_past_the_end_of_history(self):
        self.assertTrue((self.trades["exit_date"] <= self.hist["date"].max()).all())


class TestTimeSplit(unittest.TestCase):
    def test_test_period_is_last_months_and_training_is_purged(self):
        hist = _history(25000 + np.random.default_rng(0).normal(0, 100, 300).cumsum())
        trades = build_option_trades(hist, dte_choices=(2, 14), offsets_pct=(0.0,))
        train, test, cutoff = time_split(trades, test_months=3)
        self.assertTrue((test["entry_date"] >= cutoff).all())
        self.assertTrue((train["entry_date"] < cutoff).all())
        self.assertTrue((train["exit_date"] < cutoff).all())  # no trade still open at the cutoff
        self.assertGreater(len(train), len(test))


if __name__ == "__main__":
    unittest.main()
