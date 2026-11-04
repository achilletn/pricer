import numpy as np

from src.delta_hedging import hedging_pnl_distribution, simulate_hedged_pnl

S0, K, R, T, SIGMA_IMPL = 100.0, 100.0, 0.05, 1.0, 0.20
N_STEPS = 60
N_PATHS = 8000


def test_seller_profits_when_realized_vol_below_implied():
    res = hedging_pnl_distribution(S0, K, R, T, SIGMA_IMPL, sigma_real=0.10,
                                    n_steps=N_STEPS, n_paths=N_PATHS,
                                    rng=np.random.default_rng(0))
    assert res["hedging_pnl_mean"] > 0


def test_seller_loses_when_realized_vol_above_implied():
    res = hedging_pnl_distribution(S0, K, R, T, SIGMA_IMPL, sigma_real=0.30,
                                    n_steps=N_STEPS, n_paths=N_PATHS,
                                    rng=np.random.default_rng(0))
    assert res["hedging_pnl_mean"] < 0


def test_mean_pnl_near_zero_when_realized_equals_implied():
    res = hedging_pnl_distribution(S0, K, R, T, SIGMA_IMPL, sigma_real=SIGMA_IMPL,
                                    n_steps=N_STEPS, n_paths=N_PATHS,
                                    rng=np.random.default_rng(0))
    stderr = res["hedging_pnl_std"] / np.sqrt(N_PATHS)
    assert abs(res["hedging_pnl_mean"]) < 4 * stderr


def test_simulated_pnl_matches_theoretical_gamma_decomposition():
    # Le P&L reel (comptabilite cash) doit matcher la decomposition Gamma-vol
    # theorique a l'erreur de discretisation pres : c'est le garde-fou contre
    # une erreur de signe ou de comptabilite dans simulate_hedged_pnl.
    res = hedging_pnl_distribution(S0, K, R, T, SIGMA_IMPL, sigma_real=0.30,
                                    n_steps=252, n_paths=20_000,
                                    rng=np.random.default_rng(1))
    relative_diff = abs(res["hedging_pnl_mean"] - res["gamma_pnl_mean"]) / abs(res["hedging_pnl_mean"])
    assert relative_diff < 0.10


def test_discretization_error_shrinks_with_rebalancing_frequency():
    # L'ecart par trajectoire entre P&L reel et decomposition theorique vient
    # de la discretisation temporelle : il doit diminuer quand on rebalance
    # plus souvent.
    coarse_hedging, coarse_gamma = simulate_hedged_pnl(
        S0, K, R, T, SIGMA_IMPL, sigma_real=0.30, n_steps=12, n_paths=N_PATHS,
        rng=np.random.default_rng(2))
    fine_hedging, fine_gamma = simulate_hedged_pnl(
        S0, K, R, T, SIGMA_IMPL, sigma_real=0.30, n_steps=252, n_paths=N_PATHS,
        rng=np.random.default_rng(2))

    coarse_residual_std = (coarse_hedging - coarse_gamma).std()
    fine_residual_std = (fine_hedging - fine_gamma).std()
    assert fine_residual_std < coarse_residual_std
