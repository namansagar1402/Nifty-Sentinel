"""Chart helpers: hide the time when the NSE market is closed so candles sit side by side."""

from typing import Dict, List

import pandas as pd

MARKET_OPEN_HOUR = 9.25    # 09:15 IST
MARKET_CLOSE_HOUR = 15.5   # 15:30 IST


def market_closed_rangebreaks(times: pd.Series, intraday: bool, daily: bool = True) -> List[Dict]:
    """
    Plotly x-axis rangebreaks for NSE candles (times in IST, timezone-naive):
    weekends, exchange holidays (weekdays with no candle) and, for intraday candles, overnight hours.
    Weekly and monthly candles (daily=False, intraday=False) need none.
    """
    if not intraday and not daily:
        return []
    breaks: List[Dict] = [dict(bounds=["sat", "mon"])]
    if intraday:
        breaks.append(dict(bounds=[MARKET_CLOSE_HOUR, MARKET_OPEN_HOUR], pattern="hour"))

    trading_days = set(pd.to_datetime(times).dt.normalize())
    if trading_days:
        weekdays = pd.bdate_range(min(trading_days), max(trading_days))
        holidays = [d.strftime("%Y-%m-%d") for d in weekdays if d not in trading_days]
        if holidays:
            breaks.append(dict(values=holidays))
    return breaks
