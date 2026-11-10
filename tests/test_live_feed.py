import unittest

import pandas as pd

from data_feed.live_feed import compute_day_change_pct


def _candles(rows):
    return pd.DataFrame(rows, columns=["time", "Open", "Close"]).assign(time=lambda d: pd.to_datetime(d["time"]))


class TestDayChange(unittest.TestCase):
    def test_intraday_multi_day_uses_previous_session_close(self):
        # Five-day window: started at 23,600, closed yesterday at 25,000, now 25,100.
        df = _candles([
            ("2026-11-04 09:15", 23600, 23650),
            ("2026-11-09 15:25", 25010, 25000),
            ("2026-11-10 09:15", 25000, 25050),
            ("2026-11-10 10:00", 25050, 25100),
        ])
        self.assertAlmostEqual(compute_day_change_pct(df), 0.4, places=6)

    def test_daily_candles_use_previous_row_close(self):
        df = _candles([
            ("2026-11-06", 24000, 24500),
            ("2026-11-09", 24500, 25000),
            ("2026-11-10", 25000, 24750),
        ])
        self.assertAlmostEqual(compute_day_change_pct(df), -1.0, places=6)

    def test_single_session_falls_back_to_session_open(self):
        df = _candles([
            ("2026-11-10 09:15", 25000, 25050),
            ("2026-11-10 10:00", 25050, 25250),
        ])
        self.assertAlmostEqual(compute_day_change_pct(df), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
