"""Headless smoke test: the dashboard loads and every tab renders content."""
import os
import tempfile
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parent.parent / "ui" / "app.py"


_TMP_DIR = None


def setUpModule():
    """Point the app's saved portfolio at a temp file so tests never touch data/portfolio.json."""
    global _TMP_DIR
    _TMP_DIR = tempfile.TemporaryDirectory()
    os.environ["SENTINEL_PORTFOLIO_PATH"] = str(Path(_TMP_DIR.name) / "portfolio.json")


def tearDownModule():
    os.environ.pop("SENTINEL_PORTFOLIO_PATH", None)
    _TMP_DIR.cleanup()


class TestAppTabs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = AppTest.from_file(str(APP_PATH), default_timeout=180).run()

    def test_app_runs_without_exceptions(self):
        self.assertEqual([e.value for e in self.app.exception], [])

    def test_every_tab_renders_content(self):
        self.assertEqual(len(self.app.tabs), 6)
        for tab in self.app.tabs:
            with self.subTest(tab=tab.label):
                self.assertGreater(len(tab.children), 0, f"Tab '{tab.label}' is empty")



class TestDemoMode(unittest.TestCase):
    def test_demo_mode_uses_snapshot_and_says_so(self):
        at = AppTest.from_file(str(APP_PATH), default_timeout=180)
        at.session_state["demo_mode"] = True
        at.run()
        self.assertEqual([e.value for e in at.exception], [])
        html = " ".join(m.value for m in at.markdown)
        self.assertIn("● Cached", html)
        self.assertNotIn("● Live", html)


class TestAddStrategyToPortfolio(unittest.TestCase):
    def test_add_button_puts_every_leg_in_the_portfolio(self):
        at = AppTest.from_file(str(APP_PATH), default_timeout=180)
        at.session_state["demo_mode"] = True
        at.run()
        before = len(at.session_state["portfolio_manager"].positions)
        button = next(b for b in at.button if b.label.startswith("+ Add Iron Condor") or
                      b.label.startswith("+ Add 4-Leg Defined Iron Condor"))
        button.click().run()
        self.assertEqual([e.value for e in at.exception], [])
        self.assertEqual(len(at.session_state["portfolio_manager"].positions), before + 4)


class TestOptionChainControls(unittest.TestCase):
    def test_greeks_toggle_and_later_expiry_render(self):
        at = AppTest.from_file(str(APP_PATH), default_timeout=180)
        at.session_state["demo_mode"] = True
        at.run()
        at.toggle(key="chain_show_greeks").set_value(True)
        box = at.selectbox(key="chain_expiry")
        box.select_index(len(box.options) - 1).run()
        self.assertEqual([e.value for e in at.exception], [])
        text = " ".join(m.value for m in at.markdown)
        self.assertIn("% of premium", text)
        self.assertIn("Max pain", text)


class TestTradeTester(unittest.TestCase):
    def test_sell_and_later_expiry_render_with_budget_and_heatmap(self):
        at = AppTest.from_file(str(APP_PATH), default_timeout=180)
        at.session_state["demo_mode"] = True
        at.run()
        at.selectbox(key="test_act_sel").select_index(1)
        expiry = at.selectbox(key="test_expiry")
        expiry.select_index(len(expiry.options) - 1).run()
        self.assertEqual([e.value for e in at.exception], [])
        text = " ".join(m.value for m in at.markdown)
        self.assertIn("Worst case vs daily loss limit", text)
        self.assertIn("Premium collected", text)


class TestPortfolioTab(unittest.TestCase):
    def test_strategy_shows_as_group_and_close_strategy_removes_it(self):
        at = AppTest.from_file(str(APP_PATH), default_timeout=180)
        at.session_state["demo_mode"] = True
        at.run()
        groups_before = sum(b.label == "Close strategy" for b in at.button)
        add = next(b for b in at.button if b.label.startswith("+ Add 4-Leg Defined Iron Condor"))
        add.click().run()
        close = [b for b in at.button if b.label == "Close strategy"]
        self.assertEqual(len(close), groups_before + 1)
        before = len(at.session_state["portfolio_manager"].positions)
        close[0].click().run()
        self.assertEqual([e.value for e in at.exception], [])
        self.assertEqual(len(at.session_state["portfolio_manager"].positions), before - 4)
        text = " ".join(m.value for m in at.markdown)
        self.assertIn("What moves your portfolio", text)


if __name__ == "__main__":
    unittest.main()
