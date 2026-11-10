"""
Real-time market data feed connector for Nifty 50 and India VIX.
Fetches live intraday candlesticks via yfinance and generates dynamic option chains with Greeks.
"""

from typing import Dict, Any, List, Optional, Tuple
import datetime
import numpy as np
import pandas as pd
import yfinance as yf

from core.models import OptionType, OptionContract, Greeks
from core.greeks import black_scholes_price, calculate_greeks


DEFAULT_INDIA_VIX = 14.85  # typical India VIX, used only when no data source is available


def compute_day_change_pct(df: pd.DataFrame) -> float:
    """
    Today's % change versus the previous session's close (how exchanges quote it).

    Expects a candle frame with 'time', 'Open', 'Close' sorted by time.
    Falls back to the session's first Open when only one session is present.
    """
    if df.empty:
        return 0.0
    latest = float(df["Close"].iloc[-1])
    dates = pd.to_datetime(df["time"]).dt.normalize()
    earlier = df[dates < dates.iloc[-1]]
    reference = float(earlier["Close"].iloc[-1]) if not earlier.empty else float(df["Open"].iloc[0])
    return ((latest - reference) / reference) * 100.0 if reference > 0 else 0.0


class LiveDataFeed:
    """
    Connects to live market data feeds and generates real-time options analytics.
    """

    @staticmethod
    def download_candles(interval: str = "5m", period: str = "5d") -> Optional[pd.DataFrame]:
        """
        Raw Nifty 50 (^NSEI) OHLCV candles from Yahoo Finance with a naive 'time' column.
        Returns None when the feed is unreachable or empty, so callers can fall back honestly.
        """
        try:
            ticker = yf.Ticker("^NSEI")
            df = ticker.history(period=period, interval=interval)
            if df.empty:
                # Intraday is empty when the market is closed; use recent daily candles instead.
                df = ticker.history(period="1mo", interval="1d")
            if df.empty:
                return None
        except Exception as exc:
            print(f"[live_feed] Nifty candles unavailable ({interval}/{period}): {exc}")
            return None

        df = df.reset_index()
        stamp_col = "Datetime" if "Datetime" in df.columns else ("Date" if "Date" in df.columns else None)
        if stamp_col:
            df["time"] = pd.to_datetime(df[stamp_col]).dt.tz_localize(None)
        else:
            df["time"] = pd.date_range(end=datetime.datetime.now(), periods=len(df), freq="5min")
        return df[["time", "Open", "High", "Low", "Close", "Volume"]]

    @staticmethod
    def enrich_candles(df: pd.DataFrame) -> Tuple[float, float, pd.DataFrame]:
        """Adds EMA20/EMA50/VWAP and returns (spot, today's % change, candles)."""
        df = df.copy()
        df["time"] = pd.to_datetime(df["time"])
        df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
        df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
        typical = (df["High"] + df["Low"] + df["Close"]) / 3
        df["VWAP"] = (df["Volume"] * typical).cumsum() / df["Volume"].cumsum().replace(0, np.nan)
        df["VWAP"] = df["VWAP"].bfill().ffill().fillna(df["Close"])
        return round(float(df["Close"].iloc[-1]), 2), round(compute_day_change_pct(df), 2), df

    @staticmethod
    def synthetic_candles(n: int = 50, base_spot: float = 24850.0) -> pd.DataFrame:
        """Random-walk candles used only when there is neither a live feed nor a saved snapshot."""
        now = datetime.datetime.now()
        times = [now - datetime.timedelta(minutes=5 * i) for i in range(n)][::-1]
        closes = base_spot + np.cumsum(np.random.normal(0.5, 8.0, size=n))
        opens = np.roll(closes, 1)
        opens[0] = base_spot
        return pd.DataFrame({
            "time": times,
            "Open": opens,
            "High": np.maximum(opens, closes) + np.random.uniform(2, 10, size=n),
            "Low": np.minimum(opens, closes) - np.random.uniform(2, 10, size=n),
            "Close": closes,
            "Volume": np.random.randint(50000, 250000, size=n),
        })

    @classmethod
    def fetch_nifty_spot_and_candles(cls, interval: str = "5m", period: str = "5d") -> Tuple[float, float, pd.DataFrame]:
        """
        Live Nifty 50 spot and candles, or synthetic candles if the feed is down.
        Prefer data_feed.market_source.get_nifty, which also says where the data came from.
        """
        raw = cls.download_candles(interval, period)
        return cls.enrich_candles(raw if raw is not None else cls.synthetic_candles())

    @staticmethod
    def download_vix() -> Optional[float]:
        """Latest India VIX close, or None when the feed is unreachable."""
        try:
            v_hist = yf.Ticker("^INDIAVIX").history(period="5d")
            if not v_hist.empty:
                return round(float(v_hist["Close"].iloc[-1]), 2)
        except Exception as exc:
            print(f"[live_feed] India VIX unavailable: {exc}")
        return None

    @classmethod
    def fetch_india_vix(cls) -> float:
        """Live India VIX, or a typical default if the feed is down."""
        value = cls.download_vix()
        return value if value is not None else DEFAULT_INDIA_VIX

    @staticmethod
    def fetch_india_vix_history(period: str = "1y") -> pd.Series:
        """
        Daily India VIX closes, used for the IV percentile.
        Returns an empty Series when the feed is unavailable, so callers can say the value is estimated.
        """
        try:
            hist = yf.Ticker("^INDIAVIX").history(period=period, interval="1d")
            if not hist.empty:
                return hist["Close"].reset_index(drop=True)
        except Exception as exc:  # network / rate limit: degrade to "estimated"
            print(f"[live_feed] India VIX history unavailable: {exc}")
        return pd.Series(dtype=float)

    @classmethod
    def generate_live_option_chain(
        cls,
        spot_price: float,
        india_vix: float = 14.85,
        days_to_expiry: int = 5,
        num_strikes_each_side: int = 8,
        strike_step: int = 50,
        risk_free_rate: float = 0.07,
    ) -> pd.DataFrame:
        """
        Generates a comprehensive live Nifty option chain with pricing, Greeks, IV skew, and Open Interest.
        """
        atm_strike = round(spot_price / float(strike_step)) * strike_step
        start_strike = atm_strike - (num_strikes_each_side * strike_step)
        end_strike = atm_strike + (num_strikes_each_side * strike_step)

        strikes = list(range(int(start_strike), int(end_strike) + strike_step, strike_step))
        tte = max(0.001, days_to_expiry / 365.0)
        base_iv = max(0.09, india_vix / 100.0)

        chain_rows = []

        for k in strikes:
            moneyness = k / spot_price
            # Realistic volatility smile / skew
            skew_ce = base_iv + max(0.0, (moneyness - 1.0) * 0.12)
            skew_pe = base_iv + max(0.0, (1.0 - moneyness) * 0.18)

            # Call Greeks & Price
            ce_greeks = calculate_greeks(spot_price, k, tte, risk_free_rate, skew_ce, OptionType.CALL)
            pe_greeks = calculate_greeks(spot_price, k, tte, risk_free_rate, skew_pe, OptionType.PUT)

            # Open Interest distribution (higher near ATM and major round levels like 24000, 24500, 25000)
            round_bonus = 1.6 if (k % 500 == 0) else (1.25 if (k % 100 == 0) else 1.0)
            dist_factor = max(0.2, 1.0 - abs(moneyness - 1.0) * 8.0)

            ce_oi = int(120000 * dist_factor * round_bonus * np.random.uniform(0.85, 1.15))
            pe_oi = int(135000 * dist_factor * round_bonus * np.random.uniform(0.85, 1.15))

            # Determine Moneyness tag
            if k < spot_price - 25:
                ce_tag = "ITM"
                pe_tag = "OTM"
            elif k > spot_price + 25:
                ce_tag = "OTM"
                pe_tag = "ITM"
            else:
                ce_tag = "ATM"
                pe_tag = "ATM"

            chain_rows.append({
                "strike": k,
                "is_atm": (k == atm_strike),
                # Call side
                "ce_price": round(ce_greeks.price, 2),
                "ce_delta": round(ce_greeks.delta, 2),
                "ce_theta_daily": round(ce_greeks.theta, 2),
                "ce_iv_pct": round(skew_ce * 100.0, 1),
                "ce_oi": ce_oi,
                "ce_moneyness": ce_tag,
                # Put side
                "pe_price": round(pe_greeks.price, 2),
                "pe_delta": round(pe_greeks.delta, 2),
                "pe_theta_daily": round(pe_greeks.theta, 2),
                "pe_iv_pct": round(skew_pe * 100.0, 1),
                "pe_oi": pe_oi,
                "pe_moneyness": pe_tag,
            })

        return pd.DataFrame(chain_rows)
