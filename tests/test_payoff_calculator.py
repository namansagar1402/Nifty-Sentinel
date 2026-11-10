import unittest

from core.models import StrategyLeg, TradeAction, OptionType
from strategy_optimizer.payoff_calculator import PayoffCalculator


def _long_call(strike=24850.0, premium=110.0, qty=50):
    return [StrategyLeg(action=TradeAction.BUY, option_type=OptionType.CALL, strike=strike,
                        expiry_date=None, quantity=qty, premium=premium)]


class TestPayoffCalculator(unittest.TestCase):
    def test_rejects_iv_given_in_percent(self):
        # India VIX 17.42 must be passed as 0.1742, not 17.42.
        with self.assertRaises(ValueError):
            PayoffCalculator.calculate_strategy_profile(_long_call(), 25044.65, 5, 0.07, 17.42)

    def test_t0_curve_is_bounded_for_a_long_call(self):
        profile = PayoffCalculator.calculate_strategy_profile(_long_call(), 25044.65, 5, 0.07, 0.1742)
        # A long call's T+0 value can't exceed spot-move gain + time value; at +6% spot
        # (~26,550) the intrinsic gain is ~(26,550-24,850-110)*50 ≈ ₹80k.
        self.assertLess(max(profile["t0_payoff_curve"]), 100_000)
        # Loss is capped at the premium paid.
        self.assertGreaterEqual(min(profile["t0_payoff_curve"]), -110.0 * 50 - 1)


if __name__ == "__main__":
    unittest.main()
