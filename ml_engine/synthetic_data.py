import pandas as pd
import numpy as np

DEFAULT_SEED = 42


def generate_synthetic_options_data(n_samples: int = 5000, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """
    Generates realistic synthetic data for Nifty 50 long options trades.
    Features: delta, gamma, theta_burn_pct, vix, days_to_expiry, moneyness.
    Target: is_profitable (1 or 0)

    Uses its own random generator seeded with `seed`, so it never touches
    numpy's global RNG and different seeds give independent datasets.
    """
    rng = np.random.default_rng(seed)
    
    # Generate random features
    # delta: absolute delta between 0.05 and 0.95
    delta = rng.uniform(0.05, 0.95, n_samples)
    
    # gamma: peaks ATM (delta ~ 0.5)
    gamma = 0.05 * np.exp(-10 * (delta - 0.5)**2) + rng.normal(0, 0.005, n_samples)
    gamma = np.clip(gamma, 0.001, 0.1)
    
    # vix: 10 to 35
    vix = rng.normal(16, 5, n_samples)
    vix = np.clip(vix, 10, 35)
    
    # DTE: 1 to 30 days
    dte = rng.integers(1, 31, n_samples)
    
    # theta burn pct: heavily depends on DTE (low DTE = high burn)
    # e.g. burn % = (1 / DTE) * (VIX / 15) * 10%
    theta_burn_pct = (1.0 / dte) * (vix / 15.0) * 10.0 + rng.normal(0, 1, n_samples)
    theta_burn_pct = np.clip(theta_burn_pct, 0.5, 50.0)
    
    # Moneyness: Spot / Strike (Rough approx based on delta)
    moneyness = 0.8 + (delta * 0.4) 
    
    df = pd.DataFrame({
        "delta": delta,
        "gamma": gamma,
        "theta_burn_pct": theta_burn_pct,
        "vix": vix,
        "days_to_expiry": dte,
        "moneyness": moneyness
    })
    
    # Score calculation
    # Profitable when Delta is high, DTE is high, VIX is low, Theta burn is low
    score = (delta * 2.0) - (theta_burn_pct / 10.0) - (vix / 30.0) + (dte / 30.0)
    
    # Add stochastic noise
    noise = rng.normal(0, 0.5, n_samples)
    final_score = score + noise
    
    # Target: top 40% are profitable
    threshold = np.percentile(final_score, 60)
    df["is_profitable"] = (final_score >= threshold).astype(int)
    
    return df
