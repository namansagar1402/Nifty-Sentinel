"""The charcoal-and-gold design uses no emojis: this keeps them from creeping back in."""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIRS = ("ui", "core", "risk_engine", "strategy_optimizer", "trade_analyzer", "ml_engine",
               "backtesting", "data_feed", "scripts")

# Pictographs, dingbats, arrows and the variation selector that turns symbols into emojis.
# Plain typographic marks we do use (₹ · — ● ±) are outside these ranges.
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿⬀-⯿⌀-⏿️‍]")


class TestNoEmoji(unittest.TestCase):
    def test_source_files_contain_no_emojis(self):
        hits = []
        for folder in SOURCE_DIRS:
            for path in (ROOT / folder).rglob("*.py"):
                for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                    if EMOJI.search(line):
                        hits.append(f"{path.relative_to(ROOT)}:{n}: {line.strip()[:80]}")
        self.assertEqual(hits, [], "Emojis found:\n" + "\n".join(hits))


if __name__ == "__main__":
    unittest.main()
