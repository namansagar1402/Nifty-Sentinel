"""
Nifty 50 option expiry calendar.

NSE moved Nifty weekly expiries to Tuesday from September 2025; the monthly contract expires
on the last Tuesday of the month. Exchange holidays (which pull an expiry one trading day
earlier) are not modelled.
"""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import List

EXPIRY_WEEKDAY = 1  # Monday=0, so 1 = Tuesday


@dataclass(frozen=True)
class Expiry:
    expiry: date
    days_to_expiry: int
    is_monthly: bool
    label: str


def _is_last_weekday_of_month(day: date) -> bool:
    return (day + timedelta(days=7)).month != day.month


def _label(expiry: date, days: int, monthly: bool) -> str:
    when = "today" if days == 0 else ("1 day" if days == 1 else f"{days} days")
    kind = "monthly" if monthly else "weekly"
    return f"{expiry:%d %b} · {kind} · {when}"


def upcoming_expiries(today: date, weekly_count: int = 4) -> List[Expiry]:
    """The next `weekly_count` weekly expiries plus the next two monthly ones, soonest first."""
    first = today + timedelta(days=(EXPIRY_WEEKDAY - today.weekday()) % 7)
    weekly = [first + timedelta(weeks=i) for i in range(weekly_count)]

    monthly, cursor = [], first
    while len(monthly) < 2:
        if _is_last_weekday_of_month(cursor):
            monthly.append(cursor)
        cursor += timedelta(weeks=1)

    expiries = []
    for day in sorted(set(weekly + monthly)):
        days = (day - today).days
        monthly_flag = _is_last_weekday_of_month(day)
        expiries.append(Expiry(day, days, monthly_flag, _label(day, days, monthly_flag)))
    return expiries
