"""
Unit tests for Black-Scholes pricing and Greeks calculation engine.
"""

import unittest
import math
from core.models import OptionType
from core.greeks import black_scholes_price, calculate_greeks, calculate_implied_volatility


class TestGreeksEngine(unittest.TestCase):

    def test_call_put_parity_and_pricing(self):
        spot = 24500.0
        strike = 24500.0
        tte = 7.0 / 365.0
        r = 0.07
        iv = 0.15

        call_price = black_scholes_price(spot, strike, tte, r, iv, OptionType.CALL)
        put_price = black_scholes_price(spot, strike, tte, r, iv, OptionType.PUT)

        self.assertGreater(call_price, 0.0)
        self.assertGreater(put_price, 0.0)

        # Put-Call Parity: C - P = S - K * exp(-r*T)
        lhs = call_price - put_price
        rhs = spot - strike * math.exp(-r * tte)
        self.assertAlmostEqual(lhs, rhs, delta=0.5)

    def test_greeks_bounds(self):
        spot = 24500.0
        strike = 24500.0
        tte = 5.0 / 365.0
        r = 0.07
        iv = 0.14

        call_g = calculate_greeks(spot, strike, tte, r, iv, OptionType.CALL)
        put_g = calculate_greeks(spot, strike, tte, r, iv, OptionType.PUT)

        # Delta bounds
        self.assertTrue(0.0 <= call_g.delta <= 1.0)
        self.assertTrue(-1.0 <= put_g.delta <= 0.0)

        # Gamma must be positive for standard long options
        self.assertGreater(call_g.gamma, 0.0)
        self.assertGreater(put_g.gamma, 0.0)

        # Theta must be negative (time decay hurts long options)
        self.assertLess(call_g.theta, 0.0)
        self.assertLess(put_g.theta, 0.0)

        # Vega must be positive
        self.assertGreater(call_g.vega, 0.0)
        self.assertGreater(put_g.vega, 0.0)

    def test_implied_volatility_inversion(self):
        spot = 24500.0
        strike = 24500.0
        tte = 10.0 / 365.0
        r = 0.07
        target_iv = 0.18

        mkt_price = black_scholes_price(spot, strike, tte, r, target_iv, OptionType.CALL)
        recovered_iv = calculate_implied_volatility(mkt_price, spot, strike, tte, r, OptionType.CALL)

        self.assertAlmostEqual(recovered_iv, target_iv, delta=0.01)


if __name__ == "__main__":
    unittest.main()
