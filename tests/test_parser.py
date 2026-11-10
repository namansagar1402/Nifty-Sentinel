"""
Unit tests for tradebook parser and FIFO matcher.
"""

import unittest
import os
import pandas as pd
from trade_analyzer.parser import TradebookParser, parse_option_symbol
from core.models import OptionType, TradeAction


class TestTradebookParser(unittest.TestCase):

    def test_parse_option_symbol(self):
        contract1 = parse_option_symbol("NIFTY24AUG24500CE")
        self.assertEqual(contract1.underlying, "NIFTY")
        self.assertEqual(contract1.strike_price, 24500.0)
        self.assertEqual(contract1.option_type, OptionType.CALL)

        contract2 = parse_option_symbol("NIFTY2480824550PE")
        self.assertEqual(contract2.underlying, "NIFTY")
        self.assertEqual(contract2.strike_price, 24550.0)
        self.assertEqual(contract2.option_type, OptionType.PUT)

    def test_parse_sample_csv(self):
        sample_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_zerodha_trades.csv")
        self.assertTrue(os.path.exists(sample_path))

        raw_trades, closed_trades, trades_df = TradebookParser.parse_csv(sample_path)

        self.assertGreater(len(raw_trades), 0)
        self.assertGreater(len(closed_trades), 0)
        self.assertFalse(trades_df.empty)

        # Ensure gross and net pnl calculated
        self.assertTrue("gross_pnl" in trades_df.columns)
        self.assertTrue("net_pnl" in trades_df.columns)
        self.assertTrue("pnl_percent" in trades_df.columns)


if __name__ == "__main__":
    unittest.main()
