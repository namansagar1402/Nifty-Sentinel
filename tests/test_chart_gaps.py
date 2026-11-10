import unittest

import pandas as pd

from ui.chart_helpers import market_closed_rangebreaks


def _bounds(breaks):
    return [b.get("bounds") for b in breaks if "bounds" in b]


class TestMarketClosedRangebreaks(unittest.TestCase):
    def test_intraday_skips_weekends_and_overnight(self):
        times = pd.date_range("2026-11-06 09:15", "2026-11-09 15:25", freq="15min")
        times = times[(times.dayofweek < 5) & (times.hour * 60 + times.minute >= 555)
                      & (times.hour * 60 + times.minute <= 925)]
        breaks = market_closed_rangebreaks(pd.Series(times), intraday=True)
        self.assertIn(["sat", "mon"], _bounds(breaks))
        hour_breaks = [b for b in breaks if b.get("pattern") == "hour"]
        self.assertEqual(len(hour_breaks), 1)
        self.assertEqual(hour_breaks[0]["bounds"], [15.5, 9.25])

    def test_daily_skips_weekends_and_holidays_but_not_hours(self):
        days = pd.bdate_range("2026-10-26", "2026-11-06")
        days = days[days != pd.Timestamp("2026-10-30")]  # a weekday holiday with no candle
        breaks = market_closed_rangebreaks(pd.Series(days), intraday=False)
        self.assertIn(["sat", "mon"], _bounds(breaks))
        self.assertFalse(any(b.get("pattern") == "hour" for b in breaks))
        holiday_values = [v for b in breaks for v in b.get("values", [])]
        self.assertEqual(holiday_values, ["2026-10-30"])

    def test_weekly_and_monthly_candles_get_no_breaks(self):
        weeks = pd.Series(pd.date_range("2025-01-06", periods=20, freq="W-MON"))
        self.assertEqual(market_closed_rangebreaks(weeks, intraday=False, daily=False), [])


if __name__ == "__main__":
    unittest.main()
