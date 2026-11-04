import numpy as np
import pytest

from src.black_scholes import bs_call, bs_put
from src.monte_carlo import mc_price

R = 0.05
S0, K, T, SIGMA = 100.0, 100.0, 1.0, 0.2
N = 200_000


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_mc_matches_analytical_within_confidence_interval(option_type):
    rng = np.random.default_rng(42)
    result = mc_price(S0, K, R, T, SIGMA, N, option_type=option_type, rng=rng)
    analytical = bs_call(S0, K, R, T, SIGMA) if option_type == "call" else bs_put(S0, K, R, T, SIGMA)
    # Marge large (6 erreurs-std) pour eviter un test qui flake statistiquement,
    # tout en restant un vrai test (le TCL garantit ~0 depassement a cette marge).
    assert abs(result.price - analytical) < 6 * result.stderr


def test_reproducibility_with_fixed_seed():
    r1 = mc_price(S0, K, R, T, SIGMA, N, rng=np.random.default_rng(7))
    r2 = mc_price(S0, K, R, T, SIGMA, N, rng=np.random.default_rng(7))
    assert r1.price == r2.price
    assert r1.stderr == r2.stderr


def test_antithetic_reduces_variance_at_the_money():
    plain = mc_price(S0, K, R, T, SIGMA, N, antithetic=False, rng=np.random.default_rng(1))
    anti = mc_price(S0, K, R, T, SIGMA, N, antithetic=True, rng=np.random.default_rng(1))
    assert anti.stderr < plain.stderr


def test_antithetic_requires_even_n():
    with pytest.raises(ValueError):
        mc_price(S0, K, R, T, SIGMA, 3, antithetic=True, rng=np.random.default_rng(0))
