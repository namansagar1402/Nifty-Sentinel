"""
Where the dashboard's market data comes from, and an honest label for it.

Live mode:  Yahoo Finance  ->  saved snapshot  ->  simulated candles
Demo mode:  saved snapshot ->  simulated candles   (never touches the network)

Every result carries a Source so the UI can show a LIVE / CACHED / SIMULATED badge.
"""

import datetime
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Tuple

import pandas as pd

from data_feed.live_feed import DEFAULT_INDIA_VIX, LiveDataFeed

SNAPSHOT_DIR = Path(__file__).resolve().parent.parent / "data" / "market_snapshot"
META_FILE = "meta.json"
VIX_HISTORY_FILE = "vix_history.csv"

CandleFetcher = Callable[[str, str], Optional[pd.DataFrame]]

# (interval, period) pairs the app requests: the header quote plus every chart timeframe.
SNAPSHOT_TIMEFRAMES = [
    ("5m", "5d"), ("5m", "1d"), ("15m", "5d"), ("1d", "1mo"), ("1d", "3mo"),
    ("1d", "6mo"), ("1d", "1y"), ("1wk", "5y"), ("1mo", "max"),
]


class Source(str, Enum):
    LIVE = "live"
    CACHED = "cached"
    SIMULATED = "simulated"


@dataclass(frozen=True)
class NiftyQuote:
    spot: float
    change_pct: float
    candles: pd.DataFrame
    source: Source
    as_of: Optional[str]


@dataclass(frozen=True)
class VixQuote:
    value: float
    history: pd.Series
    source: Source
    as_of: Optional[str]


def _candle_file(interval: str, period: str) -> str:
    return f"candles_{interval}_{period}.csv"


def load_snapshot_meta(snapshot_dir: Path = SNAPSHOT_DIR) -> Optional[Dict]:
    path = Path(snapshot_dir) / META_FILE
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"[market_source] Snapshot metadata unreadable: {exc}")
        return None


def _load_snapshot_candles(snapshot_dir: Path, interval: str, period: str) -> Optional[pd.DataFrame]:
    path = Path(snapshot_dir) / _candle_file(interval, period)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, parse_dates=["time"])
        return df if not df.empty else None
    except (OSError, ValueError) as exc:
        print(f"[market_source] Snapshot candles unreadable ({path.name}): {exc}")
        return None


def get_nifty(
    interval: str,
    period: str,
    demo_mode: bool,
    snapshot_dir: Path = SNAPSHOT_DIR,
    fetch_candles: CandleFetcher = LiveDataFeed.download_candles,
) -> NiftyQuote:
    """Nifty 50 candles from the best available source, labelled with that source."""
    if not demo_mode:
        raw = fetch_candles(interval, period)
        if raw is not None and not raw.empty:
            spot, chg, df = LiveDataFeed.enrich_candles(raw)
            return NiftyQuote(spot, chg, df, Source.LIVE, datetime.datetime.now().strftime("%d %b %Y %H:%M"))

    meta = load_snapshot_meta(snapshot_dir)
    cached = _load_snapshot_candles(snapshot_dir, interval, period) if meta else None
    if cached is not None:
        spot, chg, df = LiveDataFeed.enrich_candles(cached)
        return NiftyQuote(spot, chg, df, Source.CACHED, meta.get("saved_at"))

    spot, chg, df = LiveDataFeed.enrich_candles(LiveDataFeed.synthetic_candles())
    return NiftyQuote(spot, chg, df, Source.SIMULATED, None)


def get_vix(
    demo_mode: bool,
    snapshot_dir: Path = SNAPSHOT_DIR,
    fetch_vix: Callable[[], Optional[float]] = LiveDataFeed.download_vix,
    fetch_vix_history: Callable[[], pd.Series] = lambda: LiveDataFeed.fetch_india_vix_history("1y"),
) -> VixQuote:
    """India VIX and its 12-month history from the best available source."""
    if not demo_mode:
        value = fetch_vix()
        if value is not None:
            return VixQuote(value, fetch_vix_history(), Source.LIVE, datetime.datetime.now().strftime("%d %b %Y %H:%M"))

    meta = load_snapshot_meta(snapshot_dir)
    if meta and meta.get("india_vix") is not None:
        hist_path = Path(snapshot_dir) / VIX_HISTORY_FILE
        history = pd.read_csv(hist_path)["vix"] if hist_path.exists() else pd.Series(dtype=float)
        return VixQuote(float(meta["india_vix"]), history, Source.CACHED, meta.get("saved_at"))

    return VixQuote(DEFAULT_INDIA_VIX, pd.Series(dtype=float), Source.SIMULATED, None)


def save_snapshot(
    snapshot_dir: Path,
    combos: Iterable[Tuple[str, str]],
    fetch_candles: CandleFetcher = LiveDataFeed.download_candles,
    fetch_vix: Callable[[], Optional[float]] = LiveDataFeed.download_vix,
    fetch_vix_history: Callable[[], pd.Series] = lambda: LiveDataFeed.fetch_india_vix_history("1y"),
) -> Dict:
    """Downloads live data once and writes it to disk for offline demos. Returns the metadata."""
    snapshot_dir = Path(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    saved, missing = [], []
    for interval, period in combos:
        df = fetch_candles(interval, period)
        if df is None or df.empty:
            missing.append(f"{interval}/{period}")
            continue
        df.to_csv(snapshot_dir / _candle_file(interval, period), index=False)
        saved.append(f"{interval}/{period}")

    history = fetch_vix_history()
    if history is not None and not history.empty:
        pd.DataFrame({"vix": history.to_numpy()}).to_csv(snapshot_dir / VIX_HISTORY_FILE, index=False)

    meta = {
        "saved_at": datetime.datetime.now().strftime("%d %b %Y %H:%M"),
        "india_vix": fetch_vix(),
        "timeframes": saved,
        "missing_timeframes": missing,
    }
    (snapshot_dir / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
