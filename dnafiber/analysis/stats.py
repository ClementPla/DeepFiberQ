import numpy as np
from scipy import stats

# Derrick, Toher & White, 2017


def partover_test(
    x1=None,
    x2=None,
    x3=None,
    x4=None,
    var_equal=False,
    mu=0.0,
    alternative="two.sided",
    conf_level=None,
):
    """
    Partially overlapping samples t-test.

    Parameters:
        x1 : array-like or None
            Unpaired sample from group 1.
        x2 : array-like or None
            Unpaired sample from group 2.
        x3 : array-like or None
            Paired sample from group 1.
        x4 : array-like or None
            Paired sample from group 2 (must be same length as x3).
        var_equal : bool
            Assume equal variances if True (Student-type); otherwise Welch-type.
        mu : float
            Null hypothesis difference.
        alternative : {'two.sided', 'less', 'greater'}
            Defines the alternative hypothesis.
        conf_level : float or None
            If provided, confidence interval is computed (0<conf_level<1).

    Returns:
        dict with keys:
            statistic, parameter (df), p.value, estimate, conf.int (if requested)
    """

    # Convert all non-None inputs to numpy arrays
    def arr_or_empty(x):
        return np.array(x, dtype=float) if x is not None else np.array([], dtype=float)

    x1 = arr_or_empty(x1)
    x2 = arr_or_empty(x2)
    x3 = arr_or_empty(x3)
    x4 = arr_or_empty(x4)

    # Check inputs
    if len(x3) != len(x4):
        raise ValueError("Paired observations not of same length")
    if len(x3) < 2:
        raise ValueError("Not enough paired observations")
    if len(x1) == 0 and len(x2) == 0:
        raise ValueError("Not enough vectors specified")

    # Common statistics
    xbar1 = np.mean(np.concatenate([x1, x3])) if len(x1) + len(x3) > 0 else 0.0
    xbar2 = np.mean(np.concatenate([x2, x4])) if len(x2) + len(x4) > 0 else 0.0
    estimate = xbar1 - xbar2

    n1 = len(x1) + len(x3)
    n2 = len(x2) + len(x4)
    n12 = len(x3)

    # correlation r of paired observations
    if np.std(x3, ddof=1) == 0 or np.std(x4, ddof=1) == 0:
        r = 0.0
    else:
        r = np.corrcoef(x3, x4)[0, 1]

    # Pooled or Welch-type standard error
    if var_equal:
        s1_sq = np.var(np.concatenate([x1, x3]), ddof=1)
        s2_sq = np.var(np.concatenate([x2, x4]), ddof=1)
        spooled = np.sqrt(((n1 - 1) * s1_sq + (n2 - 1) * s2_sq) / (n1 + n2 - 2))

        denom1 = 1 / n1 + 1 / n2
        denom2 = 2 * r * n12 / (n1 * n2)
        denom = spooled * np.sqrt(denom1 - denom2)

        statistic = (estimate - mu) / denom if denom != 0 else np.nan
        # Degrees of freedom formula from original R code
        parameter = (n12 - 1) + (
            ((len(x1) + len(x2) + len(x3) - 1) / (len(x1) + len(x2) + 2 * len(x3)))
            * (len(x1) + len(x2))
        )
    else:
        s1_sq = np.var(np.concatenate([x1, x3]), ddof=1)
        s2_sq = np.var(np.concatenate([x2, x4]), ddof=1)
        denom1 = s1_sq / n1 + s2_sq / n2
        denom2 = (
            2
            * r
            * n12
            * np.std(np.concatenate([x1, x3]), ddof=1)
            * np.std(np.concatenate([x2, x4]), ddof=1)
            / (n1 * n2)
        )
        denom = np.sqrt(denom1 - denom2)

        statistic = (estimate - mu) / denom if denom != 0 else np.nan

        # Welch-Satterthwaite approximation
        wel_numer = (s1_sq / n1 + s2_sq / n2) ** 2
        wel_denom = ((s1_sq / n1) ** 2) / (n1 - 1) + ((s2_sq / n2) ** 2) / (n2 - 1)
        welapprox = wel_numer / wel_denom
        parameter = (n12 - 1) + (
            ((welapprox - n12 + 1) / (len(x1) + len(x2) + 2 * n12))
            * (len(x1) + len(x2))
        )

    # p-value
    if np.isnan(statistic):
        p_value = 1.0
    else:
        if alternative == "less":
            p_value = stats.t.cdf(statistic, df=parameter)
        elif alternative == "greater":
            p_value = stats.t.sf(statistic, df=parameter)  # 1-cdf
        elif alternative == "two.sided":
            p_value = 2 * stats.t.sf(abs(statistic), df=parameter)
        else:
            raise ValueError("alternative must be 'two.sided', 'less' or 'greater'")

    result = {
        "statistic": statistic,
        "parameter": parameter,
        "p.value": p_value,
        "estimate": estimate,
    }

    # Confidence interval if requested
    if conf_level is not None:
        alpha = 1 - conf_level
        if alternative == "two.sided":
            tcrit = stats.t.ppf(1 - alpha / 2, df=parameter)
            lower = estimate - tcrit * denom
            upper = estimate + tcrit * denom
        elif alternative == "less":
            tcrit = stats.t.ppf(conf_level, df=parameter)
            lower = -np.inf
            upper = estimate + tcrit * denom
        elif alternative == "greater":
            tcrit = stats.t.ppf(conf_level, df=parameter)
            lower = estimate - tcrit * denom
            upper = np.inf
        result["conf.int"] = (lower, upper)

    return result
