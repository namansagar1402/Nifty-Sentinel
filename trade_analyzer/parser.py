"""
Multi-broker tradebook parser and FIFO trade-matching engine for Nifty F&O options.
Supports Zerodha, Groww, Dhan, Angel One, Upstox, and generic CSV formats.
"""

import re
import math
from datetime import datetime, date, time, timedelta
from typing import List, Dict, Tuple, Optional, Any
import pandas as pd
import numpy as np

from core.models import (
    OptionType,
    TradeAction,
    OptionContract,
    RawTrade,
    ClosedTrade,
    DailySummary,
)


def parse_option_symbol(symbol_str: str) -> OptionContract:
    """
    Parses Indian exchange option symbols into structured OptionContract.
    Examples handled:
    - NIFTY24AUG24500CE
    - NIFTY2482224500CE (Weekly: 24 = year 2024, 8 = Aug, 22 = day 22)
    - NIFTY 24500 CE 29-AUG-2024
    - NIFTY 22AUG24 24500 CE
    - NIFTY24D1924500PE (Monthly/Weekly variants)
    - NIFTY FUT / NIFTY24AUGFUT
    """
    s = symbol_str.strip().upper().replace(" ", "")

    underlying = "NIFTY"
    if "BANKNIFTY" in s:
        underlying = "BANKNIFTY"
    elif "FINNIFTY" in s:
        underlying = "FINNIFTY"
    elif "MIDCPNIFTY" in s:
        underlying = "MIDCPNIFTY"
    elif "NIFTY" in s:
        underlying = "NIFTY"
    else:
        underlying = s.split()[0] if " " in symbol_str else "NIFTY"

    # Default lot sizes
    lot_sizes = {
        "NIFTY": 50,  # 50 in standard, earlier 75/65, 50 current
        "BANKNIFTY": 15,
        "FINNIFTY": 25,
        "MIDCPNIFTY": 50,
    }
    lot_size = lot_sizes.get(underlying, 50)

    # Check for Futures
    if "FUT" in s:
        return OptionContract(
            symbol=symbol_str,
            underlying=underlying,
            strike_price=0.0,
            option_type=OptionType.FUTURES,
            lot_size=lot_size,
        )

    # Check for CE or PE
    option_type = OptionType.CALL if (s.endswith("CE") or "CE" in s) else (
        OptionType.PUT if (s.endswith("PE") or "PE" in s) else OptionType.EQUITY
    )

    # Extract strike price (look for 4-5 digit number near CE/PE)
    strike_matches = re.findall(r"(\d{4,5})(?:CE|PE)?", s)
    strike = 0.0
    if strike_matches:
        strike = float(strike_matches[-1])

    # Extract expiry if possible
    expiry = None
    # Try pattern like 24AUG / 24822 / 29-AUG-2024
    month_map = {
        "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
        "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12
    }

    # Match standard monthly pattern like 24AUG or 2024-08-29
    match_std = re.search(r"(\d{2})([A-Z]{3})", s)
    if match_std:
        yy = int("20" + match_std.group(1))
        mon_str = match_std.group(2)
        if mon_str in month_map:
            mm = month_map[mon_str]
            # Approximate last Thursday or 28th if day not in symbol
            expiry = date(yy, mm, 28)

    return OptionContract(
        symbol=symbol_str,
        underlying=underlying,
        strike_price=strike,
        option_type=option_type,
        expiry_date=expiry,
        lot_size=lot_size,
    )


def calculate_nse_charges(trade_val: float, is_option: bool = True, is_sell: bool = False) -> float:
    """
    Computes estimated standard Indian regulatory & brokerage charges for F&O trade:
    - Brokerage: ~₹20 flat per order (prorated per trade execution)
    - STT (Securities Transaction Tax): 0.1% on sell turnover for options (or 0.02% for futures)
    - Exchange Turnover Charges: 0.0505% on premium turnover (NSE)
    - GST: 18% on (Brokerage + Exchange Charges)
    - SEBI Turnover Fee: ₹10 per crore (0.0001%)
    - Stamp Duty: 0.003% on buy side
    """
    brokerage = 20.0
    stt = (trade_val * 0.001) if (is_option and is_sell) else ((trade_val * 0.0002) if (not is_option and is_sell) else 0.0)
    exchange_charges = trade_val * 0.000505
    gst = (brokerage + exchange_charges) * 0.18
    sebi_charges = trade_val * 0.000001
    stamp_duty = (trade_val * 0.00003) if (not is_sell) else 0.0

    return brokerage + stt + exchange_charges + gst + sebi_charges + stamp_duty


class TradebookParser:
    """
    Parses multi-broker CSV/Excel files into standardized RawTrade and ClosedTrade structures.
    """

    @staticmethod
    def detect_and_normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalizes column names across brokers (Zerodha, Groww, Dhan, Angel One, Upstox, Generic).
        """
        df = df.copy()
        # Clean column names
        df.columns = [str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]

        col_mapping = {}

        # Symbol mapping
        for col in ["symbol", "tradingsymbol", "instrument", "script", "scrip_name", "contract"]:
            if col in df.columns:
                col_mapping[col] = "symbol"
                break

        # Action / Trade Type mapping
        for col in ["trade_type", "type", "action", "buy_sell", "transaction_type", "side", "order_type"]:
            if col in df.columns and col != "order_type_detail":
                # Check if it contains BUY/SELL
                vals = df[col].astype(str).str.upper().unique()
                if any(v in ["BUY", "SELL", "B", "S"] for v in vals):
                    col_mapping[col] = "action"
                    break

        # Quantity mapping
        for col in ["quantity", "qty", "traded_qty", "filled_qty", "shares"]:
            if col in df.columns:
                col_mapping[col] = "quantity"
                break

        # Price mapping
        for col in ["price", "trade_price", "avg_price", "average_price", "rate", "exec_price"]:
            if col in df.columns:
                col_mapping[col] = "price"
                break

        # Date / Time mapping: prefer columns that carry the time of day (Zerodha exports both
        # trade_date and order_execution_time; the date-only one would put every trade at midnight
        # and break revenge-trade timing).
        for col in ["order_execution_time", "trade_time", "timestamp", "datetime", "fill_time", "trade_date", "date"]:
            if col in df.columns:
                col_mapping[col] = "trade_time"
                break

        # Charges mapping (if present)
        for col in ["brokerage", "charges", "total_charges", "taxes", "taxes_and_charges"]:
            if col in df.columns:
                col_mapping[col] = "charges"
                break

        df = df.rename(columns=col_mapping)
        return df

    @classmethod
    def parse_csv(cls, filepath_or_buffer: Any) -> Tuple[List[RawTrade], List[ClosedTrade], pd.DataFrame]:
        """
        Parses a tradebook file and generates matched round-trip closed trades.
        """
        if isinstance(filepath_or_buffer, pd.DataFrame):
            df = filepath_or_buffer
        elif isinstance(filepath_or_buffer, str) and filepath_or_buffer.endswith((".xlsx", ".xls")):
            df = pd.read_excel(filepath_or_buffer)
        else:
            try:
                df = pd.read_csv(filepath_or_buffer)
            except Exception:
                # Try with different separators or skip initial metadata rows (common in Zerodha/Dhan exports)
                try:
                    df = pd.read_csv(filepath_or_buffer, skiprows=1)
                except Exception:
                    df = pd.read_csv(filepath_or_buffer, sep=None, engine="python")

        df = cls.detect_and_normalize_dataframe(df)

        required_cols = ["symbol", "action", "quantity", "price"]
        for req in required_cols:
            if req not in df.columns:
                raise ValueError(f"Missing required column '{req}' in tradebook. Found columns: {list(df.columns)}")

        raw_trades: List[RawTrade] = []

        # Parse DateTime
        if "trade_time" in df.columns:
            df["parsed_time"] = pd.to_datetime(df["trade_time"], errors="coerce")
        else:
            df["parsed_time"] = pd.Timestamp.now()

        # Sort by timestamp
        df = df.sort_values(by="parsed_time").reset_index(drop=True)

        for idx, row in df.iterrows():
            sym = str(row["symbol"]).strip()
            act_str = str(row["action"]).strip().upper()
            action = TradeAction.BUY if act_str in ["BUY", "B"] else TradeAction.SELL

            qty = abs(int(float(row["quantity"])))
            price = float(row["price"])
            t_time = row["parsed_time"]
            if pd.isna(t_time):
                t_time = datetime.now()
            else:
                t_time = t_time.to_pydatetime() if hasattr(t_time, "to_pydatetime") else t_time

            charges = float(row["charges"]) if "charges" in row and not pd.isna(row["charges"]) else calculate_nse_charges(qty * price, is_sell=(action == TradeAction.SELL))

            contract = parse_option_symbol(sym)

            raw = RawTrade(
                trade_id=f"T_{idx+1}",
                symbol=sym,
                trade_time=t_time,
                action=action,
                quantity=qty,
                price=price,
                brokerage_and_charges=charges,
                option_contract=contract,
            )
            raw_trades.append(raw)

        # Match trades into ClosedTrades using FIFO
        closed_trades = cls.match_trades_fifo(raw_trades)

        # Convert closed trades to clean DataFrame for analysis
        trade_records = []
        for ct in closed_trades:
            trade_records.append({
                "trade_id": ct.trade_id,
                "symbol": ct.symbol,
                "underlying": ct.underlying,
                "option_type": ct.option_type.value,
                "strike_price": ct.strike_price,
                "direction": ct.direction,
                "entry_time": ct.entry_time,
                "exit_time": ct.exit_time,
                "date": ct.entry_time.date(),
                "quantity": ct.quantity,
                "entry_price": ct.entry_price,
                "exit_price": ct.exit_price,
                "gross_pnl": ct.gross_pnl,
                "total_charges": ct.total_charges,
                "net_pnl": ct.net_pnl,
                "pnl_percent": ct.pnl_percent,
                "holding_duration_mins": ct.holding_duration_mins,
                "is_expiry_day": ct.is_expiry_day,
                "is_revenge_trade": ct.is_revenge_trade,
                "is_overtrade": ct.is_overtrade,
                "is_martingale": ct.is_martingale,
            })
        trades_df = pd.DataFrame(trade_records)

        return raw_trades, closed_trades, trades_df

    @classmethod
    def match_trades_fifo(cls, raw_trades: List[RawTrade]) -> List[ClosedTrade]:
        """
        Matches raw BUY/SELL orders into complete round trips using FIFO (First-In, First-Out) inventory.
        Handles both Long Option Buyers (BUY first, then SELL) and Short Option Sellers (SELL first, then BUY).
        """
        closed_trades: List[ClosedTrade] = []
        # Inventory queue per symbol: list of dicts: {'action', 'qty_left', 'price', 'time', 'charges'}
        inventory: Dict[str, List[Dict[str, Any]]] = {}

        trade_counter = 1

        for raw in raw_trades:
            sym = raw.symbol
            if sym not in inventory:
                inventory[sym] = []

            curr_action = raw.action
            curr_qty = raw.quantity
            curr_price = raw.price
            curr_time = raw.trade_time
            curr_charges = raw.brokerage_and_charges
            contract = raw.option_contract or parse_option_symbol(sym)

            # Check if there are opposite orders in inventory to match against
            opposite_action = TradeAction.SELL if curr_action == TradeAction.BUY else TradeAction.BUY

            while curr_qty > 0 and len(inventory[sym]) > 0 and inventory[sym][0]["action"] == opposite_action:
                open_item = inventory[sym][0]
                matched_qty = min(curr_qty, open_item["qty_left"])

                # Determine direction:
                # If opening was BUY -> Direction is LONG
                # If opening was SELL -> Direction is SHORT
                if open_item["action"] == TradeAction.BUY:
                    direction = "LONG"
                    entry_price = open_item["price"]
                    exit_price = curr_price
                    entry_time = open_item["time"]
                    exit_time = curr_time
                    gross_pnl = (exit_price - entry_price) * matched_qty
                else:
                    direction = "SHORT"
                    entry_price = open_item["price"]
                    exit_price = curr_price
                    entry_time = open_item["time"]
                    exit_time = curr_time
                    gross_pnl = (entry_price - exit_price) * matched_qty

                # Prorated charges
                prop_open = matched_qty / open_item["orig_qty"]
                prop_close = matched_qty / raw.quantity
                matched_charges = (open_item["charges"] * prop_open) + (curr_charges * prop_close)
                net_pnl = gross_pnl - matched_charges
                pnl_pct = ((exit_price - entry_price) / entry_price * 100.0) if entry_price > 0 else 0.0
                if direction == "SHORT":
                    pnl_pct = -pnl_pct

                dur_mins = max(1.0, (exit_time - entry_time).total_seconds() / 60.0) if exit_time >= entry_time else 1.0

                # Check if entry is on a Thursday (typical Nifty weekly expiry)
                is_expiry = (entry_time.weekday() == 3)  # Thursday = 3

                closed = ClosedTrade(
                    trade_id=f"CT_{trade_counter}",
                    symbol=sym,
                    underlying=contract.underlying,
                    option_type=contract.option_type,
                    strike_price=contract.strike_price,
                    expiry_date=contract.expiry_date,
                    direction=direction,
                    entry_time=entry_time,
                    exit_time=exit_time,
                    quantity=matched_qty,
                    entry_price=round(entry_price, 2),
                    exit_price=round(exit_price, 2),
                    gross_pnl=round(gross_pnl, 2),
                    total_charges=round(matched_charges, 2),
                    net_pnl=round(net_pnl, 2),
                    pnl_percent=round(pnl_pct, 2),
                    holding_duration_mins=round(dur_mins, 1),
                    is_expiry_day=is_expiry,
                )
                closed_trades.append(closed)
                trade_counter += 1

                # Update remaining quantities
                curr_qty -= matched_qty
                open_item["qty_left"] -= matched_qty

                if open_item["qty_left"] <= 0:
                    inventory[sym].pop(0)

            # If there is remaining quantity, push it as an open inventory item
            if curr_qty > 0:
                inventory[sym].append({
                    "action": curr_action,
                    "orig_qty": curr_qty,
                    "qty_left": curr_qty,
                    "price": curr_price,
                    "time": curr_time,
                    "charges": curr_charges * (curr_qty / raw.quantity),
                })

        return closed_trades
