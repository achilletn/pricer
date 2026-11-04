import pytest

from src.black_scholes import bs_price
from src.implied_vol import implied_vol

R = 0.05
S0 = 100.0


@pytest.mark.parametrize("K,T,sigma_true,option_type", [
    (100.0, 1.0, 0.25, "call"),   # a la monnaie
    (100.0, 1.0, 0.25, "put"),
    (100.0, 0.02, 0.30, "call"),  # maturite tres courte, mais encore a la monnaie -> vega pas trop faible
    (130.0, 1.0, 0.20, "call"),   # moderement hors la monnaie
])
def test_round_trip_recovers_true_volatility(K, T, sigma_true, option_type):
    """Cas bien conditionnes (vega non-negligeable) : sigma est identifiable
    avec precision a partir du prix."""
    market_price = bs_price(S0, K, R, T, sigma_true, option_type)
    result = implied_vol(market_price, S0, K, R, T, option_type)
    assert result.converged
    assert abs(result.sigma - sigma_true) < 1e-4
    assert result.n_iter < 100


@pytest.mark.parametrize("K,T,sigma_true,option_type", [
    (300.0, 1.0, 0.20, "call"),  # tres hors la monnaie -> vega proche de 0
    (10.0, 1.0, 0.20, "call"),   # tres dans la monnaie -> vega proche de 0
])
def test_low_vega_case_still_recovers_a_consistent_price(K, T, sigma_true, option_type):
    """Cas mal conditionnes (vega quasi nulle) : le solveur doit
    quand meme converger sur le PRIX (grace au filet de securite par
    bissection), meme si sigma en lui-meme n'est plus identifiable avec
    precision (le prix est presque plat en sigma dans cette zone)."""
    market_price = bs_price(S0, K, R, T, sigma_true, option_type)
    result = implied_vol(market_price, S0, K, R, T, option_type)
    assert result.converged
    recovered_price = bs_price(S0, K, R, T, result.sigma, option_type)
    assert abs(recovered_price - market_price) < 1e-6


def test_rejects_price_outside_no_arbitrage_bounds():
    with pytest.raises(ValueError):
        implied_vol(price_market=-5.0, S0=S0, K=100.0, r=R, T=1.0)
