import unittest
from datetime import date

import pandas as pd

from core.expiry import upcoming_expiries
from data_feed.live_feed import LiveDataFeed


class TestUpcomingExpiries(unittest.TestCase):
    def test_on_expiry_tuesday_today_is_first_with_zero_days(self):
        exp = upcoming_expiries(date(2026, 11, 10))  # a Tuesday
        self.assertEqual(exp[0].expiry, date(2026, 11, 10))
        self.assertEqual(exp[0].days_to_expiry, 0)
        self.assertIn("today", exp[0].label)

    def test_mid_week_starts_from_next_tuesday(self):
        exp = upcoming_expiries(date(2026, 11, 11))  # Wednesday
        self.assertEqual(exp[0].expiry, date(2026, 11, 17))
        self.assertEqual(exp[0].days_to_expiry, 6)

    def test_last_tuesday_of_month_is_marked_monthly(self):
        exp = {e.expiry: e for e in upcoming_expiries(date(2026, 11, 10))}
        self.assertTrue(exp[date(2026, 11, 24)].is_monthly)
        self.assertFalse(exp[date(2026, 11, 17)].is_monthly)
        self.assertIn("monthly", exp[date(2026, 11, 24)].label)

    def test_includes_next_month_end_and_is_sorted_unique(self):
        exp = upcoming_expiries(date(2026, 11, 10), weekly_count=4)
        dates = [e.expiry for e in exp]
        self.assertEqual(dates, sorted(set(dates)))
        self.assertIn(date(2026, 12, 29), dates)  # last Tuesday of December
        self.assertTrue(all(d.weekday() == 1 for d in dates))


class TestSimulatedOpenInterest(unittest.TestCase):
    def test_same_inputs_give_same_open_interest(self):
        a = LiveDataFeed.generate_live_option_chain(25044.65, 17.42, days_to_expiry=7, num_strikes_each_side=5)
        b = LiveDataFeed.generate_live_option_chain(25044.65, 17.42, days_to_expiry=7, num_strikes_each_side=5)
        pd.testing.assert_series_equal(a["ce_oi"], b["ce_oi"])
        pd.testing.assert_series_equal(a["pe_oi"], b["pe_oi"])


if __name__ == "__main__":
    unittest.main()
