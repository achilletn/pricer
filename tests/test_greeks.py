from src.greeks import analytical_greeks, mc_greeks

R = 0.05
S0, K, T, SIGMA = 100.0, 100.0, 1.0, 0.2
N = 1_000_000


def test_analytical_vs_numerical_greeks_agree():
    analytical = analytical_greeks(S0, K, R, T, SIGMA, "call")
    numerical = mc_greeks(S0, K, R, T, SIGMA, N, "call", seed=123)

    assert abs(numerical["delta"] - analytical["delta"]) < 0.01
    assert abs(numerical["vega"] - analytical["vega"]) < 0.5
    assert abs(numerical["theta"] - analytical["theta"]) < 0.5
    assert abs(numerical["rho"] - analytical["rho"]) < 0.5
    # Gamma = derivee seconde -> plus bruitee (differences finies d'ordre 2), tolerance plus large
    assert abs(numerical["gamma"] - analytical["gamma"]) < 0.01


def test_put_call_delta_relation():
    call = analytical_greeks(S0, K, R, T, SIGMA, "call")
    put = analytical_greeks(S0, K, R, T, SIGMA, "put")
    assert abs((call["delta"] - put["delta"]) - 1.0) < 1e-9


def test_put_call_vega_and_gamma_identical():
    call = analytical_greeks(S0, K, R, T, SIGMA, "call")
    put = analytical_greeks(S0, K, R, T, SIGMA, "put")
    assert abs(call["vega"] - put["vega"]) < 1e-9
    assert abs(call["gamma"] - put["gamma"]) < 1e-9


def test_gamma_is_always_positive():
    for option_type in ("call", "put"):
        g = analytical_greeks(S0, K, R, T, SIGMA, option_type)
        assert g["gamma"] > 0
