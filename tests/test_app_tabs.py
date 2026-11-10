"""Headless smoke test: the dashboard loads and every tab renders content."""
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parent.parent / "ui" / "app.py"


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


if __name__ == "__main__":
    unittest.main()
