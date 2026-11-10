import unittest

from core.models import OptionType, TradeAction
from core.option_evaluator import OptionEvaluator
from ml_engine.classifier import get_predictor
from risk_engine.trade_budget import check_trade_budget

SPOT, VIX = 25044.65, 17.42


class TestPnlGrid(unittest.TestCase):
    def setUp(self):
        self.grid = OptionEvaluator.pnl_grid(SPOT, 25050, OptionType.CALL, TradeAction.BUY,
                                             days_to_expiry=5, india_vix=VIX, entry_premium=150.0, quantity=50)

    def test_shape_rows_are_moves_columns_are_days(self):
        self.assertEqual(list(self.grid.index), [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
        self.assertEqual(self.grid.columns[0], "Today")
        self.assertEqual(self.grid.columns[-1], "Expiry")

    def test_expiry_column_is_intrinsic_value_minus_premium(self):
        target = SPOT * 1.02
        expected = (max(0.0, target - 25050) - 150.0) * 50
        self.assertAlmostEqual(self.grid.loc[2.0, "Expiry"], expected, places=0)
        self.assertAlmostEqual(self.grid.loc[-2.0, "Expiry"], -150.0 * 50, places=0)

    def test_long_call_gains_as_nifty_rises(self):
        for col in self.grid.columns:
            values = list(self.grid[col])
            self.assertEqual(values, sorted(values), col)

    def test_short_is_mirror_of_long(self):
        short = OptionEvaluator.pnl_grid(SPOT, 25050, OptionType.CALL, TradeAction.SELL,
                                         days_to_expiry=5, india_vix=VIX, entry_premium=150.0, quantity=50)
        self.assertAlmostEqual(short.loc[1.0, "Expiry"], -self.grid.loc[1.0, "Expiry"], places=6)


class TestMlSecondOpinion(unittest.TestCase):
    def test_probability_is_between_0_and_1_and_prefers_slow_decay(self):
        model = get_predictor()
        slow = model.probability_profitable(delta=0.7, gamma=0.001, theta_burn_pct=2.0, vix=15.0, dte=10)
        fast = model.probability_profitable(delta=0.2, gamma=0.001, theta_burn_pct=40.0, vix=25.0, dte=1)
        self.assertTrue(0.0 <= fast <= 1.0 and 0.0 <= slow <= 1.0)
        self.assertGreater(slow, fast)


class TestTradeBudget(unittest.TestCase):
    def test_trade_within_limit_fits(self):
        b = check_trade_budget(max_loss_inr=2000.0, daily_limit_inr=2550.0, loss_per_lot_inr=2000.0)
        self.assertTrue(b.fits)
        self.assertEqual(b.max_lots_within_limit, 1)
        self.assertAlmostEqual(b.limit_used_ratio, 2000 / 2550)

    def test_oversized_trade_says_how_many_lots_fit(self):
        b = check_trade_budget(max_loss_inr=7265.0, daily_limit_inr=2550.0, loss_per_lot_inr=7265.0)
        self.assertFalse(b.fits)
        self.assertEqual(b.max_lots_within_limit, 0)
        self.assertAlmostEqual(b.limit_used_ratio, 7265 / 2550)


if __name__ == "__main__":
    unittest.main()
