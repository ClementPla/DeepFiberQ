
from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Literal, Optional
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

SummaryStat = Literal["mean", "median"]

@dataclass
class DemingResult:
    slope: float
    intercept: float
    lambda_ratio: float

def _ensure_two_graders(df: pd.DataFrame):
    graders = pd.Index(df["Grader"].astype(str).unique())
    if len(graders) != 2:
        raise ValueError(f"Expected exactly 2 graders; found {len(graders)}: {list(graders)}")
    return str(graders[0]), str(graders[1])

def _estimate_lambda_from_within(df_value_scale: pd.DataFrame, graders) -> float:
    # df_value_scale must have columns: Type, Grader, Ratio (here "Ratio" means the chosen Value scale).
    wv = df_value_scale.groupby(["Type","Grader"])["Ratio"].var(ddof=1).reset_index()
    wv = wv.replace({np.nan: 0.0})
    vx_series = wv.loc[wv["Grader"]==graders[0], "Ratio"]
    vy_series = wv.loc[wv["Grader"]==graders[1], "Ratio"]
    vx = float(vx_series.to_numpy().mean().item()) if not vx_series.empty else 1.0
    vy = float(vy_series.to_numpy().mean().item()) if not vy_series.empty else 1.0
    vx = max(vx, 1e-12)
    vy = max(vy, 1e-12)
    return float(vy / vx)

def _deming(x, y, lam: float):
    x = np.asarray(x); y = np.asarray(y)
    xbar, ybar = x.mean(), y.mean()
    Sxx = np.var(x, ddof=1); Syy = np.var(y, ddof=1); Sxy = np.cov(x, y, ddof=1)[0,1]
    term = (Syy - lam*Sxx)
    slope = (term + np.sqrt(term**2 + 4*lam*(Sxy**2))) / (2*Sxy)
    intercept = ybar - slope*xbar
    return slope, intercept

def tost_equivalence(param, ci_low, ci_high, lower_bound, upper_bound):
    """
    Check equivalence by CI inclusion within bounds.
    Returns: (equivalent_bool, message_string)
    """
    equivalent = (ci_low >= lower_bound) and (ci_high <= upper_bound)
    txt = (f"{param:.3f} (95% CI {ci_low:.3f}–{ci_high:.3f}), "
           f"bounds [{lower_bound}, {upper_bound}] → "
           + ("Equivalent ✅" if equivalent else "Not equivalent ❌"))
    return equivalent, txt

def analyze_trend_agreement(
    df: pd.DataFrame,
    use_log: bool = True,
    summary_stat: SummaryStat = "mean",
    figures_dir: Optional[str] = "/mnt/data",
    prefix: str = "cluster_agreement",
    intercept_margin: float = 0.05,   # ± log-units
    slope_lower: float = 0.95,        # slope lower bound
    slope_upper: float = 1.05         # slope upper bound
):
    df = df.copy()
    if not set(["Ratio","Grader","Type"]).issubset(df.columns):
        missing = set(["Ratio","Grader","Type"]) - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")
    if (df["Ratio"]<=0).any():
        raise ValueError("All Ratio values must be positive for log analysis.")
    df["Grader"] = df["Grader"].astype(str)
    graders = _ensure_two_graders(df)

    if use_log:
        df["Value"] = np.log(df["Ratio"].astype(float))
        value_label = "log(Ratio)"
    else:
        df["Value"] = df["Ratio"].astype(float)
        value_label = "Ratio"

    # Summaries per cluster/grader
    summ = df.groupby(["Type","Grader"], observed=True)["Value"].agg(
        n="size", mean="mean", median="median", sd=lambda s: s.std(ddof=1)
    ).reset_index()
    if figures_dir:
        summ_out = f"{figures_dir}/{prefix}_cluster_summaries.csv"
        summ.to_csv(summ_out, index=False)

    # Wide table of cluster-level summary
    pivot = summ.pivot(index="Type", columns="Grader", values=summary_stat).rename(
        columns={graders[0]:"A", graders[1]:"B"}
    ).dropna()
    if figures_dir:
        pivot_out = f"{figures_dir}/{prefix}_cluster_{summary_stat}s_wide.csv"
        pivot.to_csv(pivot_out)

    # Correlations
    pearson_r, pearson_p = stats.pearsonr(pivot["A"], pivot["B"])
    spearman_rho, spearman_p = stats.spearmanr(pivot["A"], pivot["B"])
    kendall_tau, kendall_p = stats.kendalltau(pivot["A"], pivot["B"], variant="b")

    # OLS B~A
    X = sm.add_constant(pivot["A"].values)
    ols_model = sm.OLS(pivot["B"].values, X).fit()
    slope_ols = float(ols_model.params[1])
    intercept_ols = float(ols_model.params[0])
    ci_ols = ols_model.conf_int(alpha=0.05)
    try:
        intercept_ci = (float(ci_ols.iloc[0,0]), float(ci_ols.iloc[0,1]))
        slope_ci = (float(ci_ols.iloc[1,0]), float(ci_ols.iloc[1,1]))
    except AttributeError:
        intercept_ci = (float(ci_ols[0,0]), float(ci_ols[0,1]))
        slope_ci = (float(ci_ols[1,0]), float(ci_ols[1,1]))

    # Deming with lambda from within-cluster variances on the chosen scale
    lam = _estimate_lambda_from_within(df.rename(columns={"Value":"Ratio"}), graders)
    slope_dem, intercept_dem = _deming(pivot["A"].values, pivot["B"].values, lam=lam)

    # Individual-level interaction
    df["Grader"] = pd.Categorical(df["Grader"], categories=list(graders), ordered=False)
    formula = "Value ~ C(Type) + C(Grader) + C(Type):C(Grader)"
    ols_full = smf.ols(formula, data=df).fit()
    anova_table = sm.stats.anova_lm(ols_full, typ=2)
    inter_p = float(anova_table.loc["C(Type):C(Grader)", "PR(>F)"]) if "C(Type):C(Grader)" in anova_table.index else np.nan

    # Figures
    figs = {}
    if figures_dir:
        fig1 = plt.figure()
        ax = plt.gca()
        ax.scatter(pivot["A"], pivot["B"])
        mn = float(min(pivot["A"].min(), pivot["B"].min()))
        mx = float(max(pivot["A"].max(), pivot["B"].max()))
        ax.plot([mn, mx], [mn, mx])
        ax.plot([mn, mx], [intercept_ols + slope_ols*mn, intercept_ols + slope_ols*mx])
        ax.set_xlabel(f"A: {value_label} ({graders[0]})")
        ax.set_ylabel(f"B: {value_label} ({graders[1]})")
        ax.set_title(f"Cluster {summary_stat}s: A vs B")
        scatter_path = f"{figures_dir}/{prefix}_scatter.png"
        fig1.savefig(scatter_path, dpi=150, bbox_inches="tight")
        plt.close(fig1)
        figs["scatter"] = scatter_path

        diffs = (pivot["B"] - pivot["A"]).sort_values()
        fig2 = plt.figure()
        ax2 = plt.gca()
        ax2.bar(range(len(diffs)), diffs.values)
        ax2.set_xticks(range(len(diffs)))
        ax2.set_xticklabels(diffs.index.astype(str), rotation=90)
        ax2.set_ylabel(f"{value_label} difference (B - A)")
        ax2.set_title("Per-cluster differences")
        diff_path = f"{figures_dir}/{prefix}_diff_bar.png"
        fig2.savefig(diff_path, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        figs["diff_bar"] = diff_path

    # Text report
    lines = []
    lines.append("=== Trend Agreement Across Clusters ===")
    lines.append(f"Scale: {'log(Ratio)' if use_log else 'Ratio'} ; cluster {summary_stat}s")
    lines.append(f"Graders: A = {graders[0]} ; B = {graders[1]}")
    lines.append(f"Clusters analyzed (both present): {len(pivot)}")
    lines.append("")
    lines.append("[Correlations across clusters]")
    lines.append(f"  Pearson r = {pearson_r:.3f} (p = {pearson_p:.3g})")
    lines.append(f"  Spearman rho = {spearman_rho:.3f} (p = {spearman_p:.3g})")
    lines.append(f"  Kendall tau = {kendall_tau:.3f} (p = {kendall_p:.3g})")
    lines.append("")
    lines.append("[OLS: B on A (cluster-level)]")
    lines.append(f"  Intercept = {intercept_ols:.4g} (95% CI {intercept_ci[0]:.4g} to {intercept_ci[1]:.4g})")
    lines.append(f"  Slope     = {slope_ols:.4g} (95% CI {slope_ci[0]:.4g} to {slope_ci[1]:.4g})")
    lines.append("")
    lines.append("[Deming regression (error in both)]")
    lines.append(f"  Lambda (σ_B^2 / σ_A^2 for error) ≈ {lam:.3g}")
    lines.append(f"  Intercept = {intercept_dem:.4g}")
    lines.append(f"  Slope     = {slope_dem:.4g}")
    lines.append("")
    lines.append("[Interaction test at individual level]")
    lines.append(f"  ANOVA p-value for C(Type):C(Grader): p = {inter_p:.3g}")
    lines.append("  (Small p suggests observers differ in how they contrast clusters.)")

    # Equivalence tests (OLS), configurable bounds
    eq_int = tost_equivalence(intercept_ols, intercept_ci[0], intercept_ci[1],
                              -intercept_margin, intercept_margin)
    eq_slope = tost_equivalence(slope_ols, slope_ci[0], slope_ci[1],
                                slope_lower, slope_upper)
    lines.append("\n[Equivalence tests (OLS)]")
    lines.append(f"  Intercept: {eq_int[1]}")
    lines.append(f"  Slope: {eq_slope[1]}")

    text_report = "\n".join(lines)

    # Save
    if figures_dir:
        with open(f"{figures_dir}/{prefix}_report.txt", "w") as f:
            f.write(text_report)

    return dict(text=text_report, figures=figs, cluster_wide=pivot)
