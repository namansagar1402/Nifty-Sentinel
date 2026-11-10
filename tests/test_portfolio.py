"""
Unit tests for F&O Portfolio Manager.
"""

import unittest
from core.models import OptionType, TradeAction
from core.portfolio import PortfolioManager


class TestPortfolioManager(unittest.TestCase):

    def test_portfolio_lifecycle_and_greeks(self):
        pm = PortfolioManager()
        self.assertEqual(len(pm.positions), 0)

        # Add Bull Call Spread (Buy 24500 CE, Sell 24700 CE)
        p1 = pm.add_position(strike=24500.0, option_type=OptionType.CALL, action=TradeAction.BUY, quantity=50, entry_price=120.0)
        p2 = pm.add_position(strike=24700.0, option_type=OptionType.CALL, action=TradeAction.SELL, quantity=50, entry_price=45.0)

        self.assertEqual(len(pm.positions), 2)

        # Update MTM
        mtm = pm.update_portfolio_mtm(current_spot=24550.0, india_vix=14.5, days_to_expiry=5)

        self.assertEqual(mtm["total_positions"], 2)
        self.assertGreater(mtm["total_capital_deployed"], 0.0)
        self.assertIn("net_delta", mtm)
        self.assertIn("net_theta_per_day_inr", mtm)
        self.assertFalse(mtm["positions_dataframe"].empty)

        # Remove Position
        removed = pm.remove_position(p1.position_id)
        self.assertTrue(removed)
        self.assertEqual(len(pm.positions), 1)

        # Clear
        pm.clear_portfolio()
        self.assertEqual(len(pm.positions), 0)


if __name__ == "__main__":
    unittest.main()
