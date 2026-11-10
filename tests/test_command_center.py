import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from risk_engine.command_center import (
    build_risk_snapshot,
    compute_adx,
    compute_iv_percentile,
    compute_max_pain,
    compute_pcr,
    latest_day_breaker_inputs,
)
from trade_analyzer.parser import TradebookParser

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample_zerodha_trades.csv"


def _chain():
    return pd.DataFrame({
        "strike": [24900, 25000, 25100],
        "ce_oi": [100, 200, 300],
        "pe_oi": [300, 200, 100],
    })


def _candles(closes):
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame({
        "Open": closes - 2, "High": closes + 5, "Low": closes - 5, "Close": closes,
        "EMA20": pd.Series(closes).ewm(span=20, adjust=False).mean(),
        "EMA50": pd.Series(closes).ewm(span=50, adjust=False).mean(),
    })


class TestMarketInputs(unittest.TestCase):
    def test_pcr_is_put_oi_over_call_oi(self):
        self.assertAlmostEqual(compute_pcr(_chain()), 600 / 600)

    def test_max_pain_is_strike_with_lowest_holder_payout(self):
        # Payout at 24900: calls 0 + puts (25000-24900)*200 + (25100-24900)*100 = 40,000
        # Payout at 25000: calls 100*100 + puts 100*100 = 20,000  <- minimum
        # Payout at 25100: calls 200*100 + 100*200 = 40,000
        self.assertEqual(compute_max_pain(_chain()), 25000)

    def test_adx_is_high_for_a_steady_trend_and_low_for_chop(self):
        trend = compute_adx(_candles(np.linspace(24000, 25000, 80)))
        chop = compute_adx(_candles(24500 + 10 * np.sin(np.arange(80))))
        self.assertGreater(trend, 25)
        self.assertLess(chop, trend)
        self.assertTrue(0 <= chop <= 100)

    def test_iv_percentile(self):
        history = pd.Series([10, 12, 14, 16, 18])
        self.assertAlmostEqual(compute_iv_percentile(15, history), 60.0)
        self.assertIsNone(compute_iv_percentile(15, pd.Series(dtype=float)))


class TestBreakerInputs(unittest.TestCase):
    def test_uses_only_the_most_recent_trading_day(self):
        df = pd.DataFrame({
            "entry_time": pd.to_datetime([
                "2026-11-09 10:00", "2026-11-10 09:30", "2026-11-10 10:00", "2026-11-10 11:00",
            ]),
            "net_pnl": [5000, 1000, -800, -700],
        })
        day_pnl, streak, count, day = latest_day_breaker_inputs(df)
        self.assertEqual(day_pnl, -500)
        self.assertEqual(streak, 2)
        self.assertEqual(count, 3)
        self.assertEqual(str(day), "2026-11-10")


class TestSnapshot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, _, cls.trades = TradebookParser.parse_csv(str(SAMPLE))
        closes = np.linspace(24800, 25050, 80)
        chain = pd.DataFrame({
            "strike": list(range(24600, 25500, 50)),
            "ce_oi": [1000] * 18, "pe_oi": [1100] * 18,
        })
        cls.snap = build_risk_snapshot(
            spot=25044.65, vix=17.42, candles=_candles(closes), chain=chain,
            trades_df=cls.trades, capital=200000.0, vix_history=pd.Series([12, 14, 16, 18, 20]),
        )

    def test_snapshot_has_every_section(self):
        for key in ("regime", "top_strategy", "strategies", "var", "monte_carlo", "breaker", "sizing", "inputs"):
            self.assertIn(key, self.snap)

    def test_regime_uses_live_inputs(self):
        self.assertEqual(self.snap["inputs"]["iv_percentile"], 60.0)
        self.assertFalse(self.snap["inputs"]["iv_percentile_estimated"])
        self.assertEqual(self.snap["regime"]["india_vix"], 17.42)

    def test_breaker_status_is_one_of_three_lights(self):
        self.assertIn(self.snap["breaker"]["light"], ("green", "amber", "red"))

    def test_monte_carlo_and_var_are_populated(self):
        self.assertEqual(self.snap["monte_carlo"]["num_simulations"], 1000)
        self.assertIn("95%", self.snap["var"]["var_metrics"])
        self.assertIsNotNone(self.snap["top_strategy"])


if __name__ == "__main__":
    unittest.main()
