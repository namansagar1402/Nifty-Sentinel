"""
Vectorized Black-Scholes Options Pricing & Greeks Engine for Nifty 50 Options.
Calculates Delta, Gamma, Theta (per day decay), Vega (per 1% IV), Rho, and Implied Volatility (IV).
"""

import math
from typing import Union, Tuple, Optional
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from core.models import OptionType, Greeks


def black_scholes_price(
    spot: float,
    strike: float,
    tte: float,  # Time to expiry in years
    risk_free_rate: float,
    iv: float,
    option_type: Union[OptionType, str] = OptionType.CALL,
) -> float:
    """
    Computes Black-Scholes European Option Price.
    
    :param spot: Current Nifty Spot / Future price
    :param strike: Option Strike Price
    :param tte: Time to Expiration in years (e.g. 7 days / 365 = 0.01918)
    :param risk_free_rate: Annualized risk-free rate (e.g. 0.07 for RBI repo/T-bill rate 7%)
    :param iv: Implied Volatility as a decimal (e.g. 0.15 for 15% IV)
    :param option_type: OptionType.CALL ("CE") or OptionType.PUT ("PE")
    :return: Theoretical Option Price
    """
    opt_val = option_type.value if hasattr(option_type, "value") else str(option_type)
    is_call = str(opt_val).upper() in ["CE", "CALL"] or "CE" in str(option_type).upper() or "CALL" in str(option_type).upper()

    if tte <= 0:
        if is_call:
            return max(0.0, spot - strike)
        else:
            return max(0.0, strike - spot)

    if iv <= 0:
        iv = 1e-6

    d1 = (math.log(spot / strike) + (risk_free_rate + 0.5 * iv ** 2) * tte) / (iv * math.sqrt(tte))
    d2 = d1 - iv * math.sqrt(tte)

    if is_call:
        price = spot * norm.cdf(d1) - strike * math.exp(-risk_free_rate * tte) * norm.cdf(d2)
    else:
        price = strike * math.exp(-risk_free_rate * tte) * norm.cdf(-d2) - spot * norm.cdf(-d1)

    return max(0.0, price)


def calculate_greeks(
    spot: float,
    strike: float,
    tte: float,  # in years
    risk_free_rate: float,
    iv: float,
    option_type: Union[OptionType, str] = OptionType.CALL,
) -> Greeks:
    """
    Computes all standard Greeks for an option.
    - Theta is annualized divided by 365 (daily decay in ₹ per share/unit).
    - Vega is scaled to 1% move in IV (0.01 change in volatility).
    - Rho is scaled to 1% move in interest rate (0.01 change).
    """
    opt_val = option_type.value if hasattr(option_type, "value") else str(option_type)
    is_call = str(opt_val).upper() in ["CE", "CALL"] or "CE" in str(option_type).upper() or "CALL" in str(option_type).upper()

    if tte <= 0:
        price = max(0.0, spot - strike) if is_call else max(0.0, strike - spot)
        delta = 1.0 if (is_call and spot > strike) else (-1.0 if (not is_call and spot < strike) else 0.0)
        return Greeks(price=price, delta=delta, gamma=0.0, theta=0.0, vega=0.0, rho=0.0, iv=iv)

    if iv <= 0:
        iv = 1e-6

    sqrt_t = math.sqrt(tte)
    d1 = (math.log(spot / strike) + (risk_free_rate + 0.5 * iv ** 2) * tte) / (iv * sqrt_t)
    d2 = d1 - iv * sqrt_t

    pdf_d1 = norm.pdf(d1)
    cdf_d1 = norm.cdf(d1)
    cdf_d2 = norm.cdf(d2)
    cdf_neg_d1 = norm.cdf(-d1)
    cdf_neg_d2 = norm.cdf(-d2)
    discount = math.exp(-risk_free_rate * tte)

    # Price
    if is_call:
        price = spot * cdf_d1 - strike * discount * cdf_d2
        delta = cdf_d1
        theta_annual = -(spot * pdf_d1 * iv) / (2 * sqrt_t) - risk_free_rate * strike * discount * cdf_d2
        rho = (strike * tte * discount * cdf_d2) / 100.0
    else:
        price = strike * discount * cdf_neg_d2 - spot * cdf_neg_d1
        delta = cdf_d1 - 1.0
        theta_annual = -(spot * pdf_d1 * iv) / (2 * sqrt_t) + risk_free_rate * strike * discount * cdf_neg_d2
        rho = (-strike * tte * discount * cdf_neg_d2) / 100.0

    gamma = pdf_d1 / (spot * iv * sqrt_t)
    theta_daily = theta_annual / 365.0  # Daily decay in points
    vega_1pct = (spot * sqrt_t * pdf_d1) / 100.0  # Point change per 1% change in IV

    return Greeks(
        price=max(0.0, price),
        delta=delta,
        gamma=gamma,
        theta=theta_daily,
        vega=vega_1pct,
        rho=rho,
        iv=iv,
    )


def calculate_implied_volatility(
    market_price: float,
    spot: float,
    strike: float,
    tte: float,
    risk_free_rate: float = 0.07,
    option_type: Union[OptionType, str] = OptionType.CALL,
    initial_guess: float = 0.20,
    tol: float = 1e-4,
    max_iter: int = 100,
) -> float:
    """
    Computes Implied Volatility (IV) given market option premium using Brent's method / Newton solver.
    """
    if market_price <= 0.0 or tte <= 0:
        return 0.0

    opt_val = option_type.value if hasattr(option_type, "value") else str(option_type)
    is_call = str(opt_val).upper() in ["CE", "CALL"] or "CE" in str(option_type).upper() or "CALL" in str(option_type).upper()
    intrinsic = max(0.0, spot - strike) if is_call else max(0.0, strike - spot)

    if market_price < intrinsic:
        return 0.001

    # Objective function
    def obj(sigma):
        return black_scholes_price(spot, strike, tte, risk_free_rate, sigma, option_type) - market_price

    # Try Brent method between 0.1% and 500% IV
    try:
        iv = brentq(obj, 1e-4, 5.0, xtol=tol, maxiter=max_iter)
        return float(iv)
    except Exception:
        # Fallback to Newton-Raphson
        sigma = initial_guess
        for _ in range(max_iter):
            g = calculate_greeks(spot, strike, tte, risk_free_rate, sigma, option_type)
            diff = g.price - market_price
            if abs(diff) < tol:
                return sigma
            # Vega is derivative w.r.t sigma (unscaled, multiply 100)
            vega_raw = g.vega * 100.0
            if abs(vega_raw) < 1e-8:
                break
            sigma = sigma - diff / vega_raw
            if sigma <= 0:
                sigma = 0.01
            elif sigma > 5.0:
                sigma = 5.0
                break
        return float(sigma)
