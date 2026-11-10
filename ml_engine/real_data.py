"""
Training trades built from real Nifty 50 and India VIX daily history.

Every trading day we "buy" calls and puts at a few strikes around spot for a few expiries,
priced with Black-Scholes at that day's close and India VIX, then hold each to expiry and
label it by what Nifty actually did: profitable if the option's intrinsic value at expiry
beat the premium paid. Features use only information available on the entry day.
"""

import logging
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

logger = logging.getLogger(__name__)

HISTORY_FILE = "history_nifty_vix_1d.csv"
LOT_SIZE = 50
RISK_FREE_RATE = 0.07
STRIKE_STEP = 50
DEFAULT_DTE_CHOICES = (2, 7, 14)                     # calendar days to expiry
DEFAULT_OFFSETS_PCT = (-2.0, -1.0, 0.0, 1.0, 2.0)    # strike distance from spot
THETA_BURN_CLIP = (0.5, 50.0)                        # same range as the simulated training data


def load_history(snapshot_dir: Path, allow_download: bool = True) -> pd.DataFrame:
    """Daily Nifty close and India VIX (columns date, close, vix) from the snapshot, downloading if missing."""
    path = Path(snapshot_dir) / HISTORY_FILE
    if path.exists():
        return pd.read_csv(path, parse_dates=["date"])
    if not allow_download:
        return pd.DataFrame(columns=["date", "close", "vix"])
    hist = download_history()
    if not hist.empty:
        path.parent.mkdir(parents=True, exist_ok=True)
        hist.to_csv(path, index=False)
    return hist


def download_history(period: str = "5y") -> pd.DataFrame:
    """Daily ^NSEI closes joined with ^INDIAVIX closes; empty if the feed is unreachable."""
    try:
        import yfinance as yf
        nifty = yf.Ticker("^NSEI").history(period=period, interval="1d")["Close"]
        vix = yf.Ticker("^INDIAVIX").history(period=period, interval="1d")["Close"]
    except Exception as exc:  # network / rate limit
        logger.error("Could not download Nifty/VIX history: %s", exc)
        return pd.DataFrame(columns=["date", "close", "vix"])
    joined = pd.concat({"close": nifty, "vix": vix}, axis=1, join="inner").dropna()
    joined.index = pd.to_datetime(joined.index).tz_localize(None).normalize()
    return joined.rename_axis("date").reset_index()


def _black_scholes(spot, strike, years, vol, is_call):
    """Vectorised Black-Scholes price, delta, gamma and daily theta (₹ per unit per calendar day)."""
    sqrt_t = np.sqrt(years)
    d1 = (np.log(spot / strike) + (RISK_FREE_RATE + 0.5 * vol ** 2) * years) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    disc = np.exp(-RISK_FREE_RATE * years)
    call = spot * norm.cdf(d1) - strike * disc * norm.cdf(d2)
    put = strike * disc * norm.cdf(-d2) - spot * norm.cdf(-d1)
    price = np.where(is_call, call, put)
    delta = np.where(is_call, norm.cdf(d1), norm.cdf(d1) - 1.0)
    gamma = norm.pdf(d1) / (spot * vol * sqrt_t)
    common = -spot * norm.pdf(d1) * vol / (2 * sqrt_t)
    theta_year = np.where(is_call, common - RISK_FREE_RATE * strike * disc * norm.cdf(d2),
                          common + RISK_FREE_RATE * strike * disc * norm.cdf(-d2))
    return price, delta, gamma, theta_year / 365.0


def build_option_trades(history: pd.DataFrame, dte_choices: Iterable[int] = DEFAULT_DTE_CHOICES,
                        offsets_pct: Iterable[float] = DEFAULT_OFFSETS_PCT) -> pd.DataFrame:
    """One row per (entry day, call/put, strike, expiry) with model features, real P&L and label."""
    hist = history.sort_values("date").reset_index(drop=True)
    dates = hist["date"].to_numpy()
    closes = hist["close"].to_numpy(dtype=float)
    last_date = dates[-1]

    rows = []
    for i, entry in enumerate(dates):
        for dte in dte_choices:
            expiry = entry + np.timedelta64(int(dte), "D")
            if expiry > last_date:
                continue
            j = int(np.searchsorted(dates, expiry, side="right")) - 1  # last trading day on/before expiry
            if j <= i:
                continue
            for offset in offsets_pct:
                strike = round(closes[i] * (1 + offset / 100.0) / STRIKE_STEP) * STRIKE_STEP
                for opt in ("CE", "PE"):
                    rows.append((i, j, dte, strike, opt))
    if not rows:
        return pd.DataFrame()

    idx = np.array([r[0] for r in rows])
    exit_idx = np.array([r[1] for r in rows])
    dte = np.array([r[2] for r in rows], dtype=float)
    strike = np.array([r[3] for r in rows], dtype=float)
    is_call = np.array([r[4] == "CE" for r in rows])

    spot = closes[idx]
    vix = hist["vix"].to_numpy(dtype=float)[idx]
    vol = np.maximum(0.08, vix / 100.0)
    price, delta, gamma, theta = _black_scholes(spot, strike, dte / 365.0, vol, is_call)
    price = np.maximum(price, 0.5)

    exit_close = closes[exit_idx]
    intrinsic = np.where(is_call, np.maximum(0.0, exit_close - strike), np.maximum(0.0, strike - exit_close))
    pnl = (intrinsic - price) * LOT_SIZE

    trades = pd.DataFrame({
        "entry_date": pd.to_datetime(dates[idx]),
        "exit_date": pd.to_datetime(dates[exit_idx]),
        "option_type": np.where(is_call, "CE", "PE"),
        "strike": strike,
        "spot": spot,
        "premium": price,
        "exit_value": intrinsic,
        "pnl_inr": pnl,
        "is_profitable": (pnl > 0).astype(int),
        # Model features (same names and ranges as the simulated data)
        "delta": np.abs(delta),
        "gamma": gamma,
        "theta_burn_pct": np.clip(np.abs(theta) / price * 100.0, *THETA_BURN_CLIP),
        "vix": vix,
        "days_to_expiry": dte,
        "moneyness": 0.8 + 0.4 * np.abs(delta),
    })
    return trades.sort_values(["entry_date", "days_to_expiry", "strike", "option_type"]).reset_index(drop=True)


def time_split(trades: pd.DataFrame, test_months: int = 6) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Train on older trades, test on the last `test_months`; drop training trades still open at the cutoff."""
    cutoff = trades["entry_date"].max() - pd.DateOffset(months=test_months)
    test = trades[trades["entry_date"] >= cutoff]
    train = trades[(trades["entry_date"] < cutoff) & (trades["exit_date"] < cutoff)]
    return train.reset_index(drop=True), test.reset_index(drop=True), cutoff
