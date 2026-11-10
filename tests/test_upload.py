import io
import unittest
from pathlib import Path

import pandas as pd

from trade_analyzer.parser import TradebookParser
from trade_analyzer.upload import MAX_UPLOAD_BYTES, load_tradebook_bytes

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample_zerodha_trades.csv"


class TestUploadLoader(unittest.TestCase):
    def test_valid_zerodha_csv_loads(self):
        result = load_tradebook_bytes(SAMPLE.read_bytes(), "tradebook.csv")
        self.assertIsNone(result.error)
        self.assertGreater(len(result.trades), 0)
        self.assertEqual(result.source_name, "tradebook.csv")

    def test_excel_file_loads(self):
        buf = io.BytesIO()
        pd.read_csv(SAMPLE).to_excel(buf, index=False)
        result = load_tradebook_bytes(buf.getvalue(), "groww.xlsx")
        self.assertIsNone(result.error)
        self.assertGreater(len(result.trades), 0)

    def test_rejects_unsupported_file_type(self):
        result = load_tradebook_bytes(b"%PDF-1.4", "statement.pdf")
        self.assertIsNone(result.trades)
        self.assertIn("CSV or Excel", result.error)

    def test_rejects_empty_and_oversized_files(self):
        self.assertIn("empty", load_tradebook_bytes(b"", "a.csv").error)
        self.assertIn("5 MB", load_tradebook_bytes(b"x" * (MAX_UPLOAD_BYTES + 1), "a.csv").error)

    def test_missing_columns_gives_friendly_message(self):
        result = load_tradebook_bytes(b"name,amount\nfoo,1\n", "bank.csv")
        self.assertIsNone(result.trades)
        self.assertIn("symbol", result.error)
        self.assertNotIn("Traceback", result.error)

    def test_no_completed_round_trips(self):
        csv = b"symbol,trade_type,quantity,price,order_execution_time\nNIFTY24AUG24500CE,BUY,50,100,2024-08-01 09:30:00\n"
        result = load_tradebook_bytes(csv, "open_only.csv")
        self.assertIsNone(result.trades)
        self.assertIn("round-trip", result.error)


class TestParserKeepsTimeOfDay(unittest.TestCase):
    def test_prefers_execution_time_over_date_only_column(self):
        _, _, trades = TradebookParser.parse_csv(str(SAMPLE))
        first_entry = pd.Timestamp(trades["entry_time"].iloc[0])
        self.assertNotEqual((first_entry.hour, first_entry.minute), (0, 0))


if __name__ == "__main__":
    unittest.main()
