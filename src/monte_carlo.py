"""Pricer Monte-Carlo pour options europeennes.

Simule S_T directement via la solution exacte du GBM (pas de discretisation
d'Euler) : pour une option europeenne, seul le prix terminal compte.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class MCResult:
    price: float
    stderr: float
    ci95: float  # demi-largeur de l'intervalle (pas la largeur totale)
    n: int

    @property
    def ci_low(self):
        return self.price - self.ci95

    @property
    def ci_high(self):
        return self.price + self.ci95


def terminal_price_from_z(S0, r, sigma, T, Z):
    """Separee de `simulate_terminal_price` pour pouvoir reutiliser les memes
    tirages Z d'un appel a l'autre (greeks par differences finies, voir greeks.py)."""
    return S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)


def simulate_terminal_price(S0, r, sigma, T, n, rng, antithetic=False):
    """Si antithetic=True, n doit etre pair (paire Z / -Z par tirage)."""
    if antithetic:
        if n % 2 != 0:
            raise ValueError("n doit etre pair pour l'antithetic sampling")
        half = rng.standard_normal(n // 2)
        Z = np.concatenate([half, -half])
    else:
        Z = rng.standard_normal(n)
    return terminal_price_from_z(S0, r, sigma, T, Z)


def mc_price(S0, K, r, T, sigma, n, option_type="call", antithetic=False, rng=None):
    """Prix Monte-Carlo avec erreur-std et IC 95% (theoreme central limite)."""
    if rng is None:
        rng = np.random.default_rng()

    ST = simulate_terminal_price(S0, r, sigma, T, n, rng, antithetic=antithetic)
    if option_type == "call":
        payoff = np.maximum(ST - K, 0.0)
    elif option_type == "put":
        payoff = np.maximum(K - ST, 0.0)
    else:
        raise ValueError(f"option_type doit etre 'call' ou 'put', recu {option_type!r}")

    discount = np.exp(-r * T)
    if antithetic:
        # i.i.d. par paire (Z, -Z), pas par tirage brut : payoff[:half] et
        # payoff[half:] sont correlees par construction.
        half = n // 2
        pair_means = 0.5 * (payoff[:half] + payoff[half:])
        price = discount * pair_means.mean()
        stderr = discount * pair_means.std(ddof=1) / np.sqrt(half)
    else:
        price = discount * payoff.mean()
        stderr = discount * payoff.std(ddof=1) / np.sqrt(n)

    return MCResult(price=price, stderr=stderr, ci95=1.96 * stderr, n=n)
