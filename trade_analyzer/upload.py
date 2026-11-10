"""
Safe loader for user-uploaded broker tradebooks (CSV or Excel).

Turns every failure into a short, user-facing message instead of a stack trace,
so the dashboard can show it next to the upload button.
"""

import io
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from trade_analyzer.parser import TradebookParser

MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB is far beyond any retail tradebook
ALLOWED_EXTENSIONS = (".csv", ".xlsx", ".xls")
SUPPORTED_BROKERS = "Zerodha, Groww, Dhan, Angel One, Upstox or any CSV"
REQUIRED_COLUMNS_HINT = ("symbol (or tradingsymbol), trade_type (BUY/SELL), quantity, price, "
                         "and a time column such as order_execution_time")


@dataclass(frozen=True)
class TradebookLoad:
    trades: Optional[pd.DataFrame]
    error: Optional[str]
    source_name: str


def _fail(name: str, message: str) -> TradebookLoad:
    return TradebookLoad(trades=None, error=message, source_name=name)


def _read_table(data: bytes, filename: str) -> pd.DataFrame:
    if filename.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(data))
    try:
        return pd.read_csv(io.BytesIO(data))
    except pd.errors.ParserError:
        # Some broker exports use ';' or tabs; let pandas sniff the separator.
        return pd.read_csv(io.BytesIO(data), sep=None, engine="python")


def load_tradebook_bytes(data: bytes, filename: str) -> TradebookLoad:
    """Validates and parses an uploaded tradebook into closed round-trip trades."""
    name = filename or "uploaded file"
    if not name.lower().endswith(ALLOWED_EXTENSIONS):
        return _fail(name, "Upload a CSV or Excel tradebook (.csv, .xlsx or .xls).")
    if not data:
        return _fail(name, "That file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        return _fail(name, "That file is larger than 5 MB. Export a shorter date range from your broker.")

    try:
        table = _read_table(data, name)
    except ImportError:
        return _fail(name, "Excel support isn't installed here. Export the tradebook as CSV instead.")
    except Exception:
        return _fail(name, "Couldn't read that file as a table. Check it opens in Excel, then try again.")

    if table.empty:
        return _fail(name, "That file has no rows.")

    try:
        _, closed, trades = TradebookParser.parse_csv(table)
    except ValueError as exc:
        if "Missing required column" in str(exc):
            return _fail(name, f"This doesn't look like a broker tradebook. It needs columns for {REQUIRED_COLUMNS_HINT}.")
        return _fail(name, "Some rows have values that aren't valid numbers or dates. Check quantity and price.")
    except Exception:
        return _fail(name, "Couldn't understand the trades in that file. Check it's an options tradebook.")

    if not closed or trades.empty:
        return _fail(name, "No completed round-trip trades found: every buy needs a matching sell (or vice versa).")
    return TradebookLoad(trades=trades, error=None, source_name=name)
