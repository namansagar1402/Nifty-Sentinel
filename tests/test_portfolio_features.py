import json
import tempfile
import unittest
from pathlib import Path

from core.models import OptionType, StrategyLeg, TradeAction
from core.portfolio import PortfolioManager, greeks_in_words, position_alert

SPOT, VIX = 25044.65, 17.42


def _condor_legs():
    return [
        StrategyLeg(TradeAction.SELL, OptionType.PUT, 24800, None, 50, 40.0),
        StrategyLeg(TradeAction.BUY, OptionType.PUT, 24600, None, 50, 15.0),
        StrategyLeg(TradeAction.SELL, OptionType.CALL, 25300, None, 50, 35.0),
        StrategyLeg(TradeAction.BUY, OptionType.CALL, 25500, None, 50, 12.0),
    ]


class TestGroups(unittest.TestCase):
    def test_add_strategy_groups_legs_and_close_removes_them_all(self):
        pm = PortfolioManager()
        pm.add_position(24850, OptionType.CALL, TradeAction.BUY, 50, 110.0)
        group = pm.add_strategy("Iron Condor", _condor_legs())
        self.assertEqual(group, "Iron Condor #1")
        self.assertEqual(sum(p.group == group for p in pm.positions), 4)
        self.assertEqual(len({p.position_id for p in pm.positions}), 5)
        self.assertEqual(pm.remove_group(group), 4)
        self.assertEqual(len(pm.positions), 1)

    def test_second_copy_of_a_strategy_gets_a_new_number(self):
        pm = PortfolioManager()
        pm.add_strategy("Iron Condor", _condor_legs())
        self.assertEqual(pm.add_strategy("Iron Condor", _condor_legs()), "Iron Condor #2")

    def test_mtm_rows_carry_the_group(self):
        pm = PortfolioManager()
        group = pm.add_strategy("Iron Condor", _condor_legs())
        df = pm.update_portfolio_mtm(SPOT, VIX)["positions_dataframe"]
        self.assertEqual(set(df["group"]), {group})


class TestSaving(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "portfolio.json"

    def tearDown(self):
        self.tmp.cleanup()

    def test_changes_are_saved_and_reloaded(self):
        pm = PortfolioManager(storage_path=self.path)
        pm.add_position(24850, OptionType.CALL, TradeAction.BUY, 50, 110.0)
        group = pm.add_strategy("Iron Condor", _condor_legs())
        again = PortfolioManager(storage_path=self.path)
        self.assertEqual(len(again.positions), 5)
        self.assertEqual(again.positions[0].option_type, OptionType.CALL)
        self.assertEqual(again.positions[1].action, TradeAction.SELL)
        self.assertEqual(again.positions[1].group, group)
        again.remove_group(group)
        self.assertEqual(len(PortfolioManager(storage_path=self.path).positions), 1)

    def test_corrupt_file_starts_empty_without_crashing(self):
        self.path.write_text("{not json", encoding="utf-8")
        pm = PortfolioManager(storage_path=self.path)
        self.assertEqual(pm.positions, [])
        self.assertTrue(pm.load_error)

    def test_csv_export_has_one_row_per_position(self):
        pm = PortfolioManager()
        pm.add_strategy("Iron Condor", _condor_legs())
        csv = pm.to_csv(SPOT, VIX)
        self.assertEqual(len(csv.strip().splitlines()), 5)
        self.assertIn("group", csv.splitlines()[0])


class TestGreeksInWords(unittest.TestCase):
    def test_long_call_gains_on_rise_and_loses_to_time(self):
        pm = PortfolioManager()
        pm.add_position(25050, OptionType.CALL, TradeAction.BUY, 50, 150.0)
        words = greeks_in_words(pm.update_portfolio_mtm(SPOT, VIX), SPOT)
        self.assertGreater(words["move_up_1pct_inr"], 0)
        self.assertLess(words["move_down_1pct_inr"], 0)
        self.assertLess(words["theta_per_day_inr"], 0)
        self.assertIn("gain", words["delta_sentence"])
        self.assertIn("lose", words["theta_sentence"])


class TestPositionAlert(unittest.TestCase):
    def test_thresholds(self):
        self.assertEqual(position_alert(-35.0), "Stop hit")
        self.assertEqual(position_alert(-26.0), "Near stop")
        self.assertEqual(position_alert(55.0), "Target hit")
        self.assertEqual(position_alert(5.0), "OK")
        self.assertEqual(position_alert(-10.0, stop_pct=10.0), "Stop hit")


if __name__ == "__main__":
    unittest.main()
