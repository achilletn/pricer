import numpy as np
import pytest

from src.black_scholes import bs_call, bs_put

R = 0.05


@pytest.mark.parametrize("T", [0.1, 0.2, 2.5])
@pytest.mark.parametrize("sigma", [0.1, 0.4])
def test_call_put_parity(T, sigma):
    S0, K = 100.0, 120.0
    lhs = bs_call(S0, K, R, T, sigma) - bs_put(S0, K, R, T, sigma)
    rhs = S0 - K * np.exp(-R * T)
    assert abs(lhs - rhs) < 1e-10


def test_deep_in_the_money_call():
    S0, K, T, sigma = 500.0, 100.0, 1.0, 0.2
    price = bs_call(S0, K, R, T, sigma)
    intrinsic_forward = S0 - K * np.exp(-R * T)
    assert abs(price - intrinsic_forward) < 1e-10


def test_deep_out_of_the_money_call():
    S0, K, T, sigma = 10.0, 100.0, 1.0, 0.2
    assert bs_call(S0, K, R, T, sigma) < 1e-10


def test_deep_in_the_money_put():
    S0, K, T, sigma = 10.0, 500.0, 1.0, 0.2
    price = bs_put(S0, K, R, T, sigma)
    intrinsic_forward = K * np.exp(-R * T) - S0
    assert abs(price - intrinsic_forward) < 1e-10


def test_deep_out_of_the_money_put():
    S0, K, T, sigma = 500.0, 10.0, 1.0, 0.2
    assert bs_put(S0, K, R, T, sigma) < 1e-10


def test_sigma_to_zero_call():
    S0, K, T = 100.0, 120.0, 1.0
    sigma = 1e-6
    price = bs_call(S0, K, R, T, sigma)
    expected = max(0.0, S0 - K * np.exp(-R * T))
    assert abs(price - expected) < 1e-6


def test_sigma_to_zero_put():
    S0, K, T = 120.0, 100.0, 1.0
    sigma = 1e-6
    price = bs_put(S0, K, R, T, sigma)
    expected = max(0.0, K * np.exp(-R * T) - S0)
    assert abs(price - expected) < 1e-6


def test_maturity_to_zero_call():
    # T -> 0 : le prix doit converger vers le payoff intrinseque max(S0-K, 0).
    # On ne teste pas T=0 exactement (division par sigma*sqrt(T) dans d1/d2).
    S0, K, sigma = 110.0, 100.0, 0.2
    T = 1e-8
    price = bs_call(S0, K, R, T, sigma)
    assert abs(price - max(0.0, S0 - K)) < 1e-4


def test_vectorized_over_spot():
    K, T, sigma = 100.0, 1.0, 0.2
    S0 = np.array([80.0, 100.0, 120.0])
    prices = bs_call(S0, K, R, T, sigma)
    assert prices.shape == S0.shape
    # Le call est strictement croissant en S0
    assert np.all(np.diff(prices) > 0)
