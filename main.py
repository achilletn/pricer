"""Orchestrateur : demo bout-en-bout des 3 briques du projet (reduction de
variance + greeks CRN, calibration Newton-Raphson, delta-hedging), sauvegarde
la figure de delta-hedging dans figures/ et affiche un tableau de synthese
des KPI. Les verifications rigoureuses (parite call-put, cas limites, MC vs
analytique, greeks, vol implicite, hedging) vivent dans tests/ (pytest) ;
ce script est la demo, pas la suite de tests.

Usage : python main.py
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.black_scholes import bs_price
from src.monte_carlo import mc_price
from src.greeks import analytical_greeks, mc_greeks
from src.implied_vol import implied_vol
from src.delta_hedging import hedging_pnl_distribution

FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Parametres de reference utilises dans toute la demo (call europeen, ATM)
S0, K, R, T, SIGMA = 100.0, 100.0, 0.05, 1.0, 0.20


# ---------------------------------------------------------------------------
# Monte-Carlo : reduction de variance (antithetic) + greeks sans bruit (CRN)
# ---------------------------------------------------------------------------

def variance_reduction_study(N=500_000, seed=1):
    plain = mc_price(S0, K, R, T, SIGMA, N, antithetic=False, rng=np.random.default_rng(seed))
    anti = mc_price(S0, K, R, T, SIGMA, N, antithetic=True, rng=np.random.default_rng(seed))
    factor = (plain.stderr / anti.stderr) ** 2
    print(f"Monte-Carlo : reduction de variance (antithetic, ATM) = x{factor:.2f} "
          f"(stderr {plain.stderr:.5f} -> {anti.stderr:.5f})")
    return {"reduction_factor": factor}


def crn_greeks_study():
    """Greeks par differences finies sur le pricer MC, common random numbers
    (memes tirages Z pour tous les bumps) pour eliminer le bruit d'echantillonnage."""
    analytical = analytical_greeks(S0, K, R, T, SIGMA, "call")
    numerical = mc_greeks(S0, K, R, T, SIGMA, 1_000_000, "call", seed=42)
    print("Monte-Carlo : greeks analytiques vs differences finies (CRN, N=1e6, S0=K=100, T=1, sigma=0.2)")
    for greek in ["delta", "gamma", "vega", "theta", "rho"]:
        print(f"  {greek:6s} analytique={analytical[greek]:9.5f}  MC (CRN)={numerical[greek]:9.5f}")
    return {"analytical": analytical, "numerical": numerical}


# ---------------------------------------------------------------------------
# Calibration : volatilite implicite par Newton-Raphson securise
# ---------------------------------------------------------------------------

def calibration_study():
    cases = [
        ("ATM", 100.0, 1.0, 0.25, "call"),
        ("ITM modere", 80.0, 1.0, 0.25, "call"),
        ("OTM profond", 160.0, 1.0, 0.25, "call"),
        ("Maturite courte", 100.0, 0.05, 0.30, "call"),
    ]
    rows = []
    for label, k, t, sigma_true, opt in cases:
        price = bs_price(S0, k, R, t, sigma_true, opt)
        result = implied_vol(price, S0, k, R, t, opt)
        rows.append((label, sigma_true, result.sigma, result.n_iter, result.converged))
        print(f"Calibration : {label:18s} sigma_vrai={sigma_true:.4f}  sigma_retrouve={result.sigma:.4f}  "
              f"iterations={result.n_iter:3d}  converge={result.converged}")

    avg_iter = float(np.mean([r[3] for r in rows]))
    return {"rows": rows, "avg_iterations": avg_iter}


# ---------------------------------------------------------------------------
# Delta-hedging : backtest de rebalancement quotidien, attribution du P&L
# a l'ecart vol realisee / vol implicite (decomposition Gamma P&L)
# ---------------------------------------------------------------------------

def delta_hedging_study(seed=5):
    sigma_impl = SIGMA
    n_steps, n_paths = 252, 20_000

    spike = hedging_pnl_distribution(S0, K, R, T, sigma_impl, sigma_real=0.25,
                                      n_steps=n_steps, n_paths=n_paths,
                                      rng=np.random.default_rng(seed))
    print(f"Delta-hedging : vol implicite {sigma_impl:.0%} -> vol realisee 25% (spike +5pt) -> "
          f"P&L moyen du vendeur = {spike['hedging_pnl_mean']:+.3f} (std {spike['hedging_pnl_std']:.3f})")

    sigma_real_grid = np.linspace(0.10, 0.30, 11)
    pnl_means, gamma_means = [], []
    for sigma_real in sigma_real_grid:
        res = hedging_pnl_distribution(S0, K, R, T, sigma_impl, sigma_real,
                                        n_steps=n_steps, n_paths=5_000,
                                        rng=np.random.default_rng(seed))
        pnl_means.append(res["hedging_pnl_mean"])
        gamma_means.append(res["gamma_pnl_mean"])

    plt.figure(figsize=(7, 5))
    plt.plot(sigma_real_grid, pnl_means, "o-", label="P&L de hedging simule (comptabilite cash)")
    plt.plot(sigma_real_grid, gamma_means, "--", label="Decomposition theorique Gamma P&L")
    plt.axvline(sigma_impl, color="gray", linestyle=":", label=f"vol implicite = {sigma_impl:.0%}")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel("Volatilite realisee du sous-jacent")
    plt.ylabel("P&L moyen du vendeur delta-hedge")
    plt.title("P&L de delta-hedging vs vol realisee (call ATM vendu a 20% de vol implicite)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "delta_hedging_pnl.png", dpi=150)
    plt.close()

    validation = hedging_pnl_distribution(S0, K, R, T, sigma_impl, sigma_real=0.30,
                                           n_steps=n_steps, n_paths=n_paths,
                                           rng=np.random.default_rng(seed))
    relative_gap = abs(validation["hedging_pnl_mean"] - validation["gamma_pnl_mean"]) / abs(validation["hedging_pnl_mean"])
    print(f"Delta-hedging : ecart P&L simule vs decomposition theorique = {relative_gap:.2%} "
          f"(vol realisee 30%, {n_steps} pas de hedge) -> figure {FIGURES_DIR / 'delta_hedging_pnl.png'}")

    return {"spike_pnl": spike["hedging_pnl_mean"], "relative_gap": relative_gap}


# ---------------------------------------------------------------------------
# Tableau KPI final
# ---------------------------------------------------------------------------

def print_and_save_kpi_table(variance_reduction, calibration, delta_hedging):
    lines = [
        "| Metrique | Valeur mesuree |",
        "|---|---|",
        f"| Facteur de reduction de variance (antithetic, ATM) | x{variance_reduction['reduction_factor']:.2f} |",
        f"| Iterations Newton-Raphson moyennes (vol implicite) | {calibration['avg_iterations']:.1f} |",
        f"| P&L de hedging (vol realisee 25% vs implicite 20%, spike +5pt) | {delta_hedging['spike_pnl']:+.3f} |",
        f"| Ecart P&L simule vs decomposition theorique Gamma | {delta_hedging['relative_gap']:.2%} |",
    ]
    table = "\n".join(lines)
    print("\n=== Tableau KPI ===")
    print(table)
    (FIGURES_DIR / "kpi_summary.md").write_text(table + "\n", encoding="utf-8")


if __name__ == "__main__":
    variance_reduction = variance_reduction_study()
    crn_greeks_study()
    calibration = calibration_study()
    delta_hedging = delta_hedging_study()
    print_and_save_kpi_table(variance_reduction, calibration, delta_hedging)
