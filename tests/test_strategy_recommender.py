import unittest

from core.models import StrategyLeg
from strategy_optimizer.payoff_calculator import PayoffCalculator
from strategy_optimizer.strategy_recommender import StrategyRecommender

# Bullish (spot above both EMAs) with elevated VIX -> only the Bull Put Credit Spread fits.
BULLISH_HIGH_IV = dict(nifty_spot=25044.65, india_vix=17.42, iv_percentile=79.0,
                       ema_20=24988.0, ema_50=24952.0, pcr_oi=1.11, max_pain=25050.0)


class TestStrategyRecommender(unittest.TestCase):
    def test_default_returns_only_strategies_that_fit(self):
        recs = StrategyRecommender.get_recommendations(**BULLISH_HIGH_IV)
        names = [s["strategy_name"] for s in recs["recommended_strategies"]]
        self.assertEqual(names, ["Bull Put Credit Spread (OTM)"])
        self.assertTrue(all(s["fits_regime"] for s in recs["recommended_strategies"]))

    def test_include_all_returns_all_six_fitting_first(self):
        recs = StrategyRecommender.get_recommendations(include_all=True, **BULLISH_HIGH_IV)
        strategies = recs["recommended_strategies"]
        self.assertEqual(len(strategies), 6)
        fits = [s["fits_regime"] for s in strategies]
        self.assertEqual(fits, sorted(fits, reverse=True))  # all True before any False
        self.assertEqual(sum(fits), 1)
        self.assertEqual(recs["top_strategy"]["strategy_name"], "Bull Put Credit Spread (OTM)")

    def test_every_strategy_carries_real_legs_matching_its_profile(self):
        recs = StrategyRecommender.get_recommendations(include_all=True, **BULLISH_HIGH_IV)
        for s in recs["recommended_strategies"]:
            with self.subTest(strategy=s["strategy_name"]):
                self.assertTrue(s["legs"])
                self.assertTrue(all(isinstance(leg, StrategyLeg) for leg in s["legs"]))
                self.assertEqual(len(s["legs"]), len(s["legs_detail"]))
                rebuilt = PayoffCalculator.calculate_strategy_profile(
                    s["legs"], BULLISH_HIGH_IV["nifty_spot"], 5, 0.07, BULLISH_HIGH_IV["india_vix"] / 100, 50)
                self.assertAlmostEqual(rebuilt["max_loss_inr"], s["profile"]["max_loss_inr"], places=2)


if __name__ == "__main__":
    unittest.main()
