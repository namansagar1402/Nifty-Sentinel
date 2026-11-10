"""The verdict must never contradict its own numbers (decay, chance of profit)."""
import unittest

from core.models import OptionType, TradeAction
from core.option_evaluator import (
    DECAY_TRAP_PCT, MIN_TAKE_PROB_PCT, OPTIMAL_MAX_DECAY_PCT, OptionEvaluator,
)

SPOT, VIX = 25044.65, 17.42
STRIKES = range(24600, 25550, 50)
DAYS = (0.25, 1, 2, 5, 12)


def _all_buys():
    for dte in DAYS:
        for strike in STRIKES:
            for opt in (OptionType.CALL, OptionType.PUT):
                yield dte, strike, opt


class TestHealthVerdict(unittest.TestCase):
    def test_itm_call_on_expiry_day_is_not_optimal(self):
        ev = OptionEvaluator.evaluate_option_health(SPOT, 24950, OptionType.CALL, TradeAction.BUY, 0.25, VIX)
        self.assertGreater(ev["daily_theta_burn_pct"], OPTIMAL_MAX_DECAY_PCT)
        self.assertNotEqual(ev["health_rating"], "OPTIMAL")

    def test_deep_itm_call_with_time_left_is_still_optimal(self):
        ev = OptionEvaluator.evaluate_option_health(SPOT, 24800, OptionType.CALL, TradeAction.BUY, 5, VIX)
        self.assertEqual(ev["health_rating"], "OPTIMAL")

    def test_optimal_never_has_heavy_decay(self):
        for dte, strike, opt in _all_buys():
            ev = OptionEvaluator.evaluate_option_health(SPOT, strike, opt, TradeAction.BUY, dte, VIX)
            if ev["health_rating"] == "OPTIMAL":
                with self.subTest(dte=dte, strike=strike, opt=opt.value):
                    self.assertLessEqual(ev["daily_theta_burn_pct"], OPTIMAL_MAX_DECAY_PCT)

    def test_decay_trap_is_always_hazardous(self):
        for dte, strike, opt in _all_buys():
            ev = OptionEvaluator.evaluate_option_health(SPOT, strike, opt, TradeAction.BUY, dte, VIX)
            if ev["daily_theta_burn_pct"] > DECAY_TRAP_PCT:
                with self.subTest(dte=dte, strike=strike, opt=opt.value):
                    self.assertEqual(ev["health_rating"], "HAZARDOUS")

    def test_summary_never_calls_heavy_decay_manageable(self):
        for dte, strike, opt in _all_buys():
            ev = OptionEvaluator.evaluate_option_health(SPOT, strike, opt, TradeAction.BUY, dte, VIX)
            if ev["daily_theta_burn_pct"] > OPTIMAL_MAX_DECAY_PCT:
                with self.subTest(dte=dte, strike=strike, opt=opt.value):
                    self.assertNotIn("manageable", ev["summary"])


class TestTakeDecision(unittest.TestCase):
    def test_take_needs_a_real_chance_of_profit(self):
        for dte, strike, opt in _all_buys():
            res = OptionEvaluator.test_trade_scenarios(SPOT, strike, opt, TradeAction.BUY, dte, VIX)
            if res["decision"] == "TAKE":
                with self.subTest(dte=dte, strike=strike, opt=opt.value):
                    self.assertGreaterEqual(res["health"]["prob_of_profit_pct"], MIN_TAKE_PROB_PCT)
                    self.assertNotEqual(res["health"]["health_rating"], "HAZARDOUS")

    def test_expiry_day_itm_call_is_dropped(self):
        res = OptionEvaluator.test_trade_scenarios(SPOT, 24950, OptionType.CALL, TradeAction.BUY, 0.25, VIX)
        self.assertEqual(res["decision"], "DROP")


if __name__ == "__main__":
    unittest.main()
