"""Volatilite implicite par Newton-Raphson, avec bissection en filet de securite.

Newton seul diverge quand le vega est proche de 0 (options tres ITM/OTM, T
tres petit) : le pas sigma - diff/vega explose ou sort du domaine sigma > 0.
Le prix etant strictement croissant en sigma, un bracket [lo, hi] encadrant
le prix marche reste toujours valide et peut se resserrer par bissection ;
on l'utilise pour retomber sur un pas de bissection quand Newton sort du
bracket ou que le vega est trop faible.
"""

from dataclasses import dataclass

from src.black_scholes import bs_price
from src.greeks import analytical_greeks


@dataclass
class ImpliedVolResult:
    sigma: float
    n_iter: int
    converged: bool


def implied_vol(price_market, S0, K, r, T, option_type="call",
                 bracket=(1e-6, 5.0), tol=1e-8, max_iter=100):
    lo, hi = bracket
    f_lo = bs_price(S0, K, r, T, lo, option_type) - price_market
    f_hi = bs_price(S0, K, r, T, hi, option_type) - price_market
    if f_lo > 0 or f_hi < 0:
        raise ValueError(
            f"price_market={price_market} hors de portee pour sigma dans {bracket} "
            f"(prix a lo={f_lo + price_market:.6g}, a hi={f_hi + price_market:.6g})"
        )

    sigma = 0.5 * (lo + hi)
    vega_floor = 1e-8

    for i in range(1, max_iter + 1):
        price = bs_price(S0, K, r, T, sigma, option_type)
        diff = price - price_market
        if abs(diff) < tol:
            return ImpliedVolResult(sigma=sigma, n_iter=i, converged=True)

        if diff > 0:
            hi = sigma
        else:
            lo = sigma

        vega = analytical_greeks(S0, K, r, T, sigma, option_type)["vega"]
        newton_sigma = sigma - diff / vega if vega > vega_floor else None

        if newton_sigma is not None and lo < newton_sigma < hi:
            sigma = newton_sigma
        else:
            sigma = 0.5 * (lo + hi)  # pas de bissection : vega trop faible ou Newton hors bracket

    return ImpliedVolResult(sigma=sigma, n_iter=max_iter, converged=False)
