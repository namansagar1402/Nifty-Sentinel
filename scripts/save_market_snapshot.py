"""
Save today's market data for offline demos.

Run this the day before a presentation (with internet):
    python scripts/save_market_snapshot.py

It writes data/market_snapshot/, which the dashboard's Demo mode uses with no network.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_feed.market_source import SNAPSHOT_DIR, SNAPSHOT_TIMEFRAMES, save_snapshot  # noqa: E402


def main() -> int:
    meta = save_snapshot(SNAPSHOT_DIR, SNAPSHOT_TIMEFRAMES)
    print(f"Snapshot saved to {SNAPSHOT_DIR} at {meta['saved_at']}")
    print(f"  India VIX: {meta['india_vix']}")
    print(f"  Timeframes saved: {', '.join(meta['timeframes']) or 'none'}")
    if meta["missing_timeframes"]:
        print(f"  Missing (feed returned nothing): {', '.join(meta['missing_timeframes'])}")
    if not meta["timeframes"] or meta["india_vix"] is None:
        print("Snapshot is incomplete: check your internet connection and run again.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
