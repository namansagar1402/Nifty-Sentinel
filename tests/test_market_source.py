import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from data_feed.market_source import (
    Source,
    get_nifty,
    get_vix,
    load_snapshot_meta,
    save_snapshot,
)


def _raw_candles(n=60, start=25000.0):
    closes = start + np.cumsum(np.full(n, 2.0))
    times = pd.date_range("2026-11-09 09:15", periods=n, freq="5min")
    return pd.DataFrame({
        "time": times, "Open": closes - 1, "High": closes + 3, "Low": closes - 3,
        "Close": closes, "Volume": np.full(n, 1000),
    })


def _live_ok(interval, period):
    return _raw_candles()


def _live_down(interval, period):
    return None


def _must_not_call(*_args):
    raise AssertionError("demo mode must not touch the network")


class TestNiftySourceChain(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.snap_dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _save(self):
        save_snapshot(self.snap_dir, combos=[("5m", "5d")], fetch_candles=_live_ok,
                      fetch_vix=lambda: 17.42, fetch_vix_history=lambda: pd.Series([12.0, 15.0, 18.0]))

    def test_live_data_is_labelled_live(self):
        q = get_nifty("5m", "5d", demo_mode=False, snapshot_dir=self.snap_dir, fetch_candles=_live_ok)
        self.assertEqual(q.source, Source.LIVE)
        self.assertIn("EMA20", q.candles.columns)

    def test_falls_back_to_snapshot_when_feed_is_down(self):
        self._save()
        q = get_nifty("5m", "5d", demo_mode=False, snapshot_dir=self.snap_dir, fetch_candles=_live_down)
        self.assertEqual(q.source, Source.CACHED)
        self.assertEqual(q.as_of, load_snapshot_meta(self.snap_dir)["saved_at"])
        self.assertAlmostEqual(q.spot, float(_raw_candles()["Close"].iloc[-1]), places=2)

    def test_demo_mode_never_calls_the_network(self):
        self._save()
        q = get_nifty("5m", "5d", demo_mode=True, snapshot_dir=self.snap_dir, fetch_candles=_must_not_call)
        self.assertEqual(q.source, Source.CACHED)

    def test_simulated_when_no_feed_and_no_snapshot(self):
        q = get_nifty("5m", "5d", demo_mode=False, snapshot_dir=self.snap_dir, fetch_candles=_live_down)
        self.assertEqual(q.source, Source.SIMULATED)
        self.assertGreater(len(q.candles), 0)

    def test_timeframe_missing_from_snapshot_is_simulated(self):
        self._save()
        q = get_nifty("1wk", "5y", demo_mode=True, snapshot_dir=self.snap_dir, fetch_candles=_must_not_call)
        self.assertEqual(q.source, Source.SIMULATED)


class TestVixSourceChain(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.snap_dir = Path(self.tmp.name)
        save_snapshot(self.snap_dir, combos=[], fetch_candles=_live_ok,
                      fetch_vix=lambda: 17.42, fetch_vix_history=lambda: pd.Series([12.0, 15.0, 18.0]))

    def tearDown(self):
        self.tmp.cleanup()

    def test_live_vix(self):
        v = get_vix(demo_mode=False, snapshot_dir=self.snap_dir, fetch_vix=lambda: 15.1,
                    fetch_vix_history=lambda: pd.Series([14.0]))
        self.assertEqual((v.value, v.source), (15.1, Source.LIVE))

    def test_demo_vix_comes_from_snapshot(self):
        v = get_vix(demo_mode=True, snapshot_dir=self.snap_dir, fetch_vix=_must_not_call,
                    fetch_vix_history=_must_not_call)
        self.assertEqual((v.value, v.source), (17.42, Source.CACHED))
        self.assertEqual(list(v.history), [12.0, 15.0, 18.0])

    def test_vix_feed_down_uses_snapshot(self):
        v = get_vix(demo_mode=False, snapshot_dir=self.snap_dir, fetch_vix=lambda: None,
                    fetch_vix_history=lambda: pd.Series(dtype=float))
        self.assertEqual(v.source, Source.CACHED)


if __name__ == "__main__":
    unittest.main()
