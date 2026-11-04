"""Pricer analytique Black-Scholes pour calls et puts europeens (vectorise NumPy)."""

import numpy as np
from scipy.stats import norm


def d1(S0, K, r, T, sigma):
    return (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def d2(S0, K, r, T, sigma):
    return d1(S0, K, r, T, sigma) - sigma * np.sqrt(T)


def bs_call(S0, K, r, T, sigma):
    D1, D2 = d1(S0, K, r, T, sigma), d2(S0, K, r, T, sigma)
    return S0 * norm.cdf(D1) - K * np.exp(-r * T) * norm.cdf(D2)


def bs_put(S0, K, r, T, sigma):
    D1, D2 = d1(S0, K, r, T, sigma), d2(S0, K, r, T, sigma)
    return -S0 * norm.cdf(-D1) + K * np.exp(-r * T) * norm.cdf(-D2)


def bs_price(S0, K, r, T, sigma, option_type="call"):
    """Dispatch call/put, utile pour implied_vol.py qui doit rester agnostique du type d'option."""
    if option_type == "call":
        return bs_call(S0, K, r, T, sigma)
    elif option_type == "put":
        return bs_put(S0, K, r, T, sigma)
    raise ValueError(f"option_type doit etre 'call' ou 'put', recu {option_type!r}")
