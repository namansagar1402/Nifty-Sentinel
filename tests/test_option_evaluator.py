"""
Unit tests for Option Health Evaluator and Pre-Trade Testing Sandbox.
"""

import unittest
from core.models import OptionType, TradeAction
from core.option_evaluator import OptionEvaluator


class TestOptionEvaluator(unittest.TestCase):

    def test_evaluate_option_health_optimal_vs_hazardous(self):
        spot = 24500.0

        # Safe ITM Call
        optimal_eval = OptionEvaluator.evaluate_option_health(
            spot_price=spot,
            strike_price=24300.0,  # 200 pts ITM
            option_type=OptionType.CALL,
            action=TradeAction.BUY,
            days_to_expiry=5,
            india_vix=14.5,
        )
        self.assertIn(optimal_eval["health_rating"], ["OPTIMAL", "SPECULATIVE"])
        self.assertEqual(optimal_eval["moneyness_tag"], "ITM")
        self.assertGreater(optimal_eval["delta"], 0.60)

        # Unstable Far OTM Call (Lottery Ticket)
        hazardous_eval = OptionEvaluator.evaluate_option_health(
            spot_price=spot,
            strike_price=25200.0,  # 700 pts OTM (>2.8% away)
            option_type=OptionType.CALL,
            action=TradeAction.BUY,
            days_to_expiry=3,
            india_vix=14.5,
        )
        self.assertEqual(hazardous_eval["health_rating"], "HAZARDOUS")
        self.assertEqual(hazardous_eval["moneyness_tag"], "OTM")
        self.assertLess(hazardous_eval["prob_of_profit_pct"], 25.0)

    def test_pre_trade_testing_scenarios_and_verdict(self):
        spot = 24500.0

        # Far OTM should recommend DROP
        drop_test = OptionEvaluator.test_trade_scenarios(
            spot_price=spot,
            strike_price=25200.0,
            option_type=OptionType.CALL,
            action=TradeAction.BUY,
            days_to_expiry=2,
            india_vix=14.5,
        )
        self.assertIn("DROP", drop_test["decision"])
        self.assertFalse(drop_test["scenario_table"].empty)

        # Near ATM/ITM should give scenario table
        atm_test = OptionEvaluator.test_trade_scenarios(
            spot_price=spot,
            strike_price=24500.0,
            option_type=OptionType.CALL,
            action=TradeAction.BUY,
            days_to_expiry=5,
            india_vix=14.5,
        )
        self.assertIn("decision", atm_test)
        self.assertEqual(len(atm_test["scenario_table"]), 7)


if __name__ == "__main__":
    unittest.main()
