
"""
Equivalence testing for two observers measuring ratios on a continuous scale.
Analyses are performed on the natural-log scale to respect multiplicative/ratio structure.

What it does:
1) Welch–TOST on log-means with a percent margin (default ±10%).
   -> Reports geometric means and their ratio, CI for ratio, and TOST p-value.

2) Bootstrap CI for variance ratio (s_A^2 / s_B^2) with bounds [0.8, 1.25] by default.
   -> Reports variance ratio and CI; declares "equivalent" if CI ⊆ [tau_L, tau_U].

3) Bootstrap CI for KS distance and Wasserstein-1 distance on the log scale.
   -> Non-inferiority check: distance ≤ epsilon (defaults: KS eps=0.08, W1 eps=0.10 log-units).

Usage
-----
Fill arrays x and y with your two observers' measurements (positive ratios).
Then run this script, or import and call `equivalence_report(...)`.

Example
-------
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    # Synthetic example: observer A ~ lognormal(mu=0, sigma=0.4), observer B ~ lognormal(mu=log(1.03), sigma=0.42)
    x = np.exp(rng.normal(0.0, 0.40, size=1200))
    y = np.exp(rng.normal(np.log(1.03), 0.42, size=900))
    report = equivalence_report(
        x, y,
        mean_margin_pct=10.0,
        var_ratio_bounds=(0.8, 1.25),
        ks_eps=0.08,
        w1_eps=0.10,
        n_boot=500,
        random_state=123
    )
    print(report["pretty"])
"""

import numpy as np
from scipy import stats
from typing import Tuple, Dict, Any, Optional

def _welch_t_ci(mean_diff: float, s2x: float, nx: int, s2y: float, ny: int, alpha: float = 0.05):
    se = np.sqrt(s2x/nx + s2y/ny)
    df_num = (s2x/nx + s2y/ny)**2
    df_den = (s2x**2 / (nx**2 * (nx-1))) + (s2y**2 / (ny**2 * (ny-1)))
    df = df_num / df_den
    tcrit = stats.t.ppf(1 - alpha/2, df)
    return mean_diff - tcrit*se, mean_diff + tcrit*se, se, df

def welch_tost_logmeans(x: np.ndarray, y: np.ndarray, mean_margin_pct: float = 10.0, alpha: float = 0.05):
    """
    TOST for equivalence of means on the log scale.
    H0: |mu_x - mu_y| >= delta, where delta = log(1 + mean_margin_pct/100).
    """
    lx = np.log(x)
    ly = np.log(y)
    nx, ny = len(lx), len(ly)
    mx, my = lx.mean(), ly.mean()
    vx, vy = lx.var(ddof=1), ly.var(ddof=1)
    delta = np.log(1 + mean_margin_pct/100.0)

    diff = mx - my
    L, U, se, df = _welch_t_ci(diff, vx, nx, vy, ny, alpha)

    # Two one-sided tests
    t1 = (diff - (-delta)) / se  # diff > -delta
    p1 = 1 - stats.t.cdf(t1, df)
    t2 = (delta - diff) / se     # diff < delta
    p2 = 1 - stats.t.cdf(t2, df)
    p_tost = max(p1, p2)  # conservative

    # Back-transform CI for geometric mean ratio
    gmr = np.exp(diff)
    ci_ratio = (np.exp(L), np.exp(U))

    return {
        "mx": mx, "my": my, "vx": vx, "vy": vy, "nx": nx, "ny": ny,
        "delta_log": delta,
        "diff_log": diff,
        "se_log": se, "df": df,
        "ci_log": (L, U),
        "gmr": gmr, "gmr_ci": ci_ratio,
        "p_tost": p_tost
    }

def bootstrap_ci(stat_fn, x: np.ndarray, y: np.ndarray, n_boot: int = 1000, alpha: float = 0.05, random_state: Optional[int] = None):
    rng = np.random.default_rng(random_state)
    nx, ny = len(x), len(y)
    stats_boot = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        xb = x[rng.integers(0, nx, size=nx)]
        yb = y[rng.integers(0, ny, size=ny)]
        stats_boot[b] = stat_fn(xb, yb)
    lo = np.percentile(stats_boot, 100*alpha/2)
    hi = np.percentile(stats_boot, 100*(1 - alpha/2))
    return lo, hi, stats_boot

def variance_ratio(x: np.ndarray, y: np.ndarray) -> float:
    return np.var(x, ddof=1) / np.var(y, ddof=1)

def ks_distance_log(x: np.ndarray, y: np.ndarray) -> float:
    lx, ly = np.log(x), np.log(y)
    d_stat, _ = stats.ks_2samp(lx, ly, alternative="two-sided", mode="auto")
    return float(d_stat)

def wasserstein_log(x: np.ndarray, y: np.ndarray) -> float:
    lx, ly = np.log(x), np.log(y)
    return float(stats.wasserstein_distance(lx, ly))

def equivalence_report(
    x: np.ndarray,
    y: np.ndarray,
    mean_margin_pct: float = 10.0,
    var_ratio_bounds: Tuple[float, float] = (0.8, 1.25),
    ks_eps: float = 0.08,
    w1_eps: float = 0.10,
    n_boot: int = 1000,
    alpha: float = 0.05,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Run the suite and return a dictionary with results and a pretty-printed report.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.any(x <= 0) or np.any(y <= 0):
        raise ValueError("All measurements must be positive (ratios).")

    # Summary
    summary = {
        "n_x": len(x), "n_y": len(y),
        "mean_x": float(np.mean(x)), "mean_y": float(np.mean(y)),
        "median_x": float(np.median(x)), "median_y": float(np.median(y)),
        "sd_x": float(np.std(x, ddof=1)), "sd_y": float(np.std(y, ddof=1)),
        "gmean_x": float(np.exp(np.mean(np.log(x)))), "gmean_y": float(np.exp(np.mean(np.log(y)))),
    }

    # 1) Mean equivalence (log scale)
    tost = welch_tost_logmeans(x, y, mean_margin_pct=mean_margin_pct, alpha=alpha)
    mean_equiv = tost["p_tost"] < alpha

    # 2) Variance ratio CI via bootstrap
    vr_obs = variance_ratio(x, y)
    vr_lo, vr_hi, _ = bootstrap_ci(variance_ratio, x, y, n_boot=n_boot, alpha=alpha, random_state=random_state)
    tau_L, tau_U = var_ratio_bounds
    var_equiv = (vr_lo >= tau_L) and (vr_hi <= tau_U)

    # 3) Distribution distances on log scale + bootstrap CIs
    ks_obs = ks_distance_log(x, y)
    ks_lo, ks_hi, _ = bootstrap_ci(ks_distance_log, x, y, n_boot=n_boot, alpha=alpha, random_state=random_state)
    ks_equiv = (ks_hi <= ks_eps)

    w1_obs = wasserstein_log(x, y)
    w1_lo, w1_hi, _ = bootstrap_ci(wasserstein_log, x, y, n_boot=n_boot, alpha=alpha, random_state=random_state)
    w1_equiv = (w1_hi <= w1_eps)

    pretty = []
    pretty.append("=== Equivalence Report (log scale for means/distances) ===")
    pretty.append(f"n_x = {summary['n_x']}, n_y = {summary['n_y']}")
    pretty.append(f"Arithmetic mean: {summary['mean_x']:.4g} vs {summary['mean_y']:.4g}")
    pretty.append(f"Geometric mean:  {summary['gmean_x']:.4g} vs {summary['gmean_y']:.4g}")
    pretty.append(f"SD: {summary['sd_x']:.4g} vs {summary['sd_y']:.4g}")
    pretty.append("")
    # Mean equivalence
    gmr = tost["gmr"]
    gmr_L, gmr_U = tost["gmr_ci"]
    pct_margin = mean_margin_pct
    pretty.append(f"[Mean equivalence] margin ±{pct_margin:.2f}% (GMR in [{1/(1+pct_margin/100):.4g}, {(1+pct_margin/100):.4g}] approximately)")
    pretty.append(f"  Geometric mean ratio (x/y): {gmr:.4g} (95% CI {gmr_L:.4g}–{gmr_U:.4g}); TOST p = {tost['p_tost']:.4g}")
    pretty.append(f"  Conclusion: {'EQUIVALENT' if mean_equiv else 'not equivalent'}")
    pretty.append("")
    # Variance ratio
    pretty.append(f"[Variance ratio equivalence] bounds [{tau_L:.3g}, {tau_U:.3g}]")
    pretty.append(f"  Observed s_x^2 / s_y^2 = {vr_obs:.4g} (bootstrap 95% CI {vr_lo:.4g}–{vr_hi:.4g})")
    pretty.append(f"  Conclusion: {'EQUIVALENT' if var_equiv else 'not equivalent'}")
    pretty.append("")
    # KS and Wasserstein
    pretty.append(f"[KS distance on log-values] ε = {ks_eps:.3g}")
    pretty.append(f"  Observed D = {ks_obs:.4g} (bootstrap 95% CI {ks_lo:.4g}–{ks_hi:.4g})")
    pretty.append(f"  Conclusion: {'EQUIVALENT' if ks_equiv else 'not equivalent'}")
    pretty.append("")
    pretty.append(f"[Wasserstein-1 on log-values] ε = {w1_eps:.3g} (log units)")
    pretty.append(f"  Observed W1 = {w1_obs:.4g} (bootstrap 95% CI {w1_lo:.4g}–{w1_hi:.4g})")
    pretty.append(f"  Conclusion: {'EQUIVALENT' if w1_equiv else 'not equivalent'}")
    pretty.append("")
    # Overall suggestion
    overall = mean_equiv and var_equiv and ks_equiv and w1_equiv
    pretty.append(f"OVERALL: {'EQUIVALENT within margins' if overall else 'Not all criteria met'}")
    pretty_text = "\n".join(pretty)

    return {
        "summary": summary,
        "tost": tost,
        "variance_ratio": {"obs": vr_obs, "ci": (vr_lo, vr_hi), "bounds": (tau_L, tau_U)},
        "ks": {"obs": ks_obs, "ci": (ks_lo, ks_hi), "eps": ks_eps},
        "w1": {"obs": w1_obs, "ci": (w1_lo, w1_hi), "eps": w1_eps},
        "overall_equiv": overall,
        "pretty": pretty_text
    }
