import math
import unittest

import pandas as pd

from data_feed.live_feed import compute_day_change_pct, previous_session_close
from risk_engine.expected_move import expected_move


class TestExpectedMove(unittest.TestCase):
    def test_one_day_move_scales_vix_by_root_252(self):
        band = expected_move(reference_price=25000.0, india_vix=17.42)
        expected_pct = 17.42 / math.sqrt(252)
        self.assertAlmostEqual(band.move_pct, expected_pct, places=6)
        self.assertAlmostEqual(band.high - band.low, 2 * 25000.0 * expected_pct / 100, places=4)
        self.assertAlmostEqual((band.high + band.low) / 2, 25000.0, places=6)
        self.assertAlmostEqual(band.probability_pct, 68.27, places=2)

    def test_longer_horizon_widens_with_square_root_of_time(self):
        one = expected_move(25000.0, 16.0, trading_days=1)
        five = expected_move(25000.0, 16.0, trading_days=5)
        self.assertAlmostEqual(five.move_pct / one.move_pct, math.sqrt(5), places=6)

    def test_rejects_bad_inputs(self):
        with self.assertRaises(ValueError):
            expected_move(0.0, 17.0)
        with self.assertRaises(ValueError):
            expected_move(25000.0, -1.0)

    def test_contains(self):
        band = expected_move(25000.0, 16.0)
        self.assertTrue(band.contains(25000.0))
        self.assertFalse(band.contains(26000.0))


class TestPreviousSessionClose(unittest.TestCase):
    def _df(self, rows):
        return pd.DataFrame(rows, columns=["time", "Open", "Close"]).assign(time=lambda d: pd.to_datetime(d["time"]))

    def test_uses_last_close_before_today(self):
        df = self._df([("2026-11-09 15:25", 25010, 25000), ("2026-11-10 09:15", 25000, 25050)])
        self.assertEqual(previous_session_close(df), 25000)

    def test_single_session_falls_back_to_open(self):
        df = self._df([("2026-11-10 09:15", 24990, 25050), ("2026-11-10 10:00", 25050, 25100)])
        self.assertEqual(previous_session_close(df), 24990)

    def test_day_change_still_matches(self):
        df = self._df([("2026-11-09 15:25", 25010, 25000), ("2026-11-10 10:00", 25050, 25100)])
        self.assertAlmostEqual(compute_day_change_pct(df), 0.4, places=6)


if __name__ == "__main__":
    unittest.main()
