"""Backtest de delta-hedging : attribution du P&L a l'ecart vol realisee / vol implicite.

    P&L ~ somme_t  0.5 * Gamma_t * S_t^2 * (sigma_impl^2 - sigma_realisee_locale_t^2) * dt
"""

import numpy as np

from src.black_scholes import bs_price
from src.greeks import analytical_greeks


def simulate_hedged_pnl(S0, K, r, T, sigma_impl, sigma_real, n_steps, n_paths,
                         option_type="call", mu=None, rng=None):
    """Retourne (hedging_pnl, gamma_pnl), deux tableaux (n_paths,) : le P&L reel
    du portefeuille auto-financant (comptabilite cash) et sa decomposition
    theorique en Gamma P&L (formule ci-dessus). Les deux doivent coincider a
    l'erreur de discretisation pres (voir tests/test_delta_hedging.py)."""
    if rng is None:
        rng = np.random.default_rng()
    if mu is None:
        mu = r

    dt = T / n_steps
    Z = rng.standard_normal((n_steps, n_paths))

    S = np.full(n_paths, float(S0))
    premium = bs_price(S0, K, r, T, sigma_impl, option_type)
    delta = analytical_greeks(S, K, r, T, sigma_impl, option_type)["delta"]
    bank = premium - delta * S  # vend l'option, achete le delta initial, place/emprunte le reste

    gamma_pnl = np.zeros(n_paths)

    for i in range(n_steps):
        tau = T - i * dt

        gamma = analytical_greeks(S, K, r, tau, sigma_impl, option_type)["gamma"]
        S_next = S * np.exp((mu - 0.5 * sigma_real**2) * dt + sigma_real * np.sqrt(dt) * Z[i])
        local_realized_var = ((S_next - S) / S) ** 2 / dt
        gamma_pnl += 0.5 * gamma * S**2 * (sigma_impl**2 - local_realized_var) * dt

        bank *= np.exp(r * dt)
        S = S_next

        if i < n_steps - 1:
            tau_next = T - (i + 1) * dt
            new_delta = analytical_greeks(S, K, r, tau_next, sigma_impl, option_type)["delta"]
            bank -= (new_delta - delta) * S  # rebalancement, finance par le compte cash
            delta = new_delta

    payoff = np.maximum(S - K, 0.0) if option_type == "call" else np.maximum(K - S, 0.0)
    portfolio_value = bank + delta * S
    hedging_pnl = portfolio_value - payoff

    return hedging_pnl, gamma_pnl


def hedging_pnl_distribution(S0, K, r, T, sigma_impl, sigma_real, n_steps, n_paths,
                              option_type="call", mu=None, rng=None):
    """Statistiques agregees sur n_paths trajectoires (voir simulate_hedged_pnl)."""
    hedging_pnl, gamma_pnl = simulate_hedged_pnl(
        S0, K, r, T, sigma_impl, sigma_real, n_steps, n_paths, option_type, mu, rng
    )
    return {
        "hedging_pnl_mean": hedging_pnl.mean(),
        "hedging_pnl_std": hedging_pnl.std(ddof=1),
        "gamma_pnl_mean": gamma_pnl.mean(),
        "hedging_pnl": hedging_pnl,
        "gamma_pnl": gamma_pnl,
    }
