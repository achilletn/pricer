"""Greeks d'options europeennes : formules fermees, et differences finies sur
le pricer Monte-Carlo (CRN : memes tirages Z pour S0+h et S0-h, sinon le bruit
d'echantillonnage domine l'effet du bump)."""

import numpy as np
from scipy.stats import norm

from src.black_scholes import d1, d2
from src.monte_carlo import terminal_price_from_z


def analytical_greeks(S0, K, r, T, sigma, option_type="call"):
    D1 = d1(S0, K, r, T, sigma)
    D2 = d2(S0, K, r, T, sigma)
    pdf_d1 = norm.pdf(D1)
    sqrtT = np.sqrt(T)

    gamma = pdf_d1 / (S0 * sigma * sqrtT)
    vega = S0 * pdf_d1 * sqrtT

    if option_type == "call":
        delta = norm.cdf(D1)
        theta = -(S0 * pdf_d1 * sigma) / (2 * sqrtT) - r * K * np.exp(-r * T) * norm.cdf(D2)
        rho = K * T * np.exp(-r * T) * norm.cdf(D2)
    elif option_type == "put":
        delta = norm.cdf(D1) - 1.0
        theta = -(S0 * pdf_d1 * sigma) / (2 * sqrtT) + r * K * np.exp(-r * T) * norm.cdf(-D2)
        rho = -K * T * np.exp(-r * T) * norm.cdf(-D2)
    else:
        raise ValueError(f"option_type doit etre 'call' ou 'put', recu {option_type!r}")

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def _mc_price_from_z(S0, K, r, T, sigma, Z, option_type):
    ST = terminal_price_from_z(S0, r, sigma, T, Z)
    if option_type == "call":
        payoff = np.maximum(ST - K, 0.0)
    else:
        payoff = np.maximum(K - ST, 0.0)
    return np.exp(-r * T) * payoff.mean()


def mc_greeks(S0, K, r, T, sigma, n, option_type="call",
              h_S=None, h_sigma=1e-3, h_T=1e-4, h_r=1e-4, seed=None):
    """Greeks par differences finies centrees, communs tirages Z (CRN)."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n)  # tire une seule fois, reutilise pour tous les bumps

    if h_S is None:
        h_S = 0.01 * S0

    def price(S0_=S0, K_=K, r_=r, T_=T, sigma_=sigma):
        return _mc_price_from_z(S0_, K_, r_, T_, sigma_, Z, option_type)

    p0 = price()
    delta = (price(S0_=S0 + h_S) - price(S0_=S0 - h_S)) / (2 * h_S)
    gamma = (price(S0_=S0 + h_S) - 2 * p0 + price(S0_=S0 - h_S)) / h_S**2
    vega = (price(sigma_=sigma + h_sigma) - price(sigma_=sigma - h_sigma)) / (2 * h_sigma)
    theta = -(price(T_=T + h_T) - price(T_=T - h_T)) / (2 * h_T)
    rho = (price(r_=r + h_r) - price(r_=r - h_r)) / (2 * h_r)

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}
