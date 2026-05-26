

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest
from typing import Tuple, Dict


def test_claim_frequency(
    group_a: pd.Series,
    group_b: pd.Series,
    group_a_name: str = "Group A",
    group_b_name: str = "Group B",
    alpha: float = 0.05
) -> Dict:
    """
    Z-test for difference in claim frequency between two groups.

    Claim frequency = proportion of policies with at least one claim.
    Uses a two-proportion z-test.

    Args:
        group_a       : TotalClaims series for control group
        group_b       : TotalClaims series for test group
        group_a_name  : label for group A
        group_b_name  : label for group B
        alpha         : significance level (default 0.05)

    Returns:
        dict with test results, p-value, and decision
    """
    # Convert to binary: 1 = had a claim, 0 = no claim
    freq_a = (group_a > 0).sum()   # count of claims in A
    freq_b = (group_b > 0).sum()   # count of claims in B
    n_a = len(group_a)              # total policies in A
    n_b = len(group_b)              # total policies in B

    # proportions_ztest takes [successes_A, successes_B], [n_A, n_B]
    stat, p_value = proportions_ztest(
        [freq_a, freq_b],
        [n_a, n_b],
        alternative='two-sided'   # test for ANY difference, not just one direction
    )

    claim_freq_a = freq_a / n_a
    claim_freq_b = freq_b / n_b
    diff = claim_freq_b - claim_freq_a
    pct_diff = (diff / claim_freq_a * 100) if claim_freq_a != 0 else 0

    decision = "REJECT H₀" if p_value < alpha else "FAIL TO REJECT H₀"
    significant = p_value < alpha

    return {
        "test_type":        "Z-test for Proportions",
        "kpi":              "Claim Frequency",
        "group_a":          group_a_name,
        "group_b":          group_b_name,
        "n_a":              n_a,
        "n_b":              n_b,
        "freq_a":           round(claim_freq_a, 6),
        "freq_b":           round(claim_freq_b, 6),
        "difference":       round(diff, 6),
        "pct_difference":   round(pct_diff, 2),
        "z_statistic":      round(stat, 4),
        "p_value":          round(p_value, 6),
        "alpha":            alpha,
        "decision":         decision,
        "significant":      significant,
    }


def test_claim_severity(
    group_a: pd.Series,
    group_b: pd.Series,
    group_a_name: str = "Group A",
    group_b_name: str = "Group B",
    alpha: float = 0.05
) -> Dict:
    """
    Z-test for difference in claim severity between two groups.

    Claim severity = average claim amount for policies WITH a claim.
    Only policies where TotalClaims > 0 are included.

    Args:
        group_a       : TotalClaims series for control group
        group_b       : TotalClaims series for test group
        group_a_name  : label for group A
        group_b_name  : label for group B
        alpha         : significance level (default 0.05)

    Returns:
        dict with test results, p-value, and decision
    """
    # Filter to only policies that had a claim
    # Zero-claim policies don't represent severity
    claims_a = group_a[group_a > 0]
    claims_b = group_b[group_b > 0]

    if len(claims_a) < 2 or len(claims_b) < 2:
        return {
            "test_type":        "Welch's t-test (z-test equivalent at large n)",
            "kpi":              "Claim Severity",
            "group_a":          group_a_name,
            "group_b":          group_b_name,
            "n_a":              len(claims_a),
            "n_b":              len(claims_b),
            "mean_a":           round(claims_a.mean(), 2) if len(claims_a) > 0 else 0.0,
            "mean_b":           round(claims_b.mean(), 2) if len(claims_b) > 0 else 0.0,
            "difference":       0.0,
            "pct_difference":   0.0,
            "t_statistic":      0.0,
            "p_value":          1.0,
            "alpha":            alpha,
            "decision":         "FAIL TO REJECT H₀",
            "significant":      False,
            "error":            "Insufficient claim data for severity test (less than 2 claims in one of the groups)"
        }

    # For large samples, t-test and z-test converge
    # scipy's ttest_ind is equivalent to z-test at large n
    stat, p_value = stats.ttest_ind(
        claims_a, claims_b,
        equal_var=False,    # Welch's t-test — doesn't assume equal variance
        alternative='two-sided'
    )

    mean_a = claims_a.mean()
    mean_b = claims_b.mean()
    diff = mean_b - mean_a
    pct_diff = (diff / mean_a * 100) if mean_a != 0 else 0

    decision = "REJECT H₀" if p_value < alpha else "FAIL TO REJECT H₀"

    return {
        "test_type":        "Welch's t-test (z-test equivalent at large n)",
        "kpi":              "Claim Severity",
        "group_a":          group_a_name,
        "group_b":          group_b_name,
        "n_a":              len(claims_a),
        "n_b":              len(claims_b),
        "mean_a":           round(mean_a, 2),
        "mean_b":           round(mean_b, 2),
        "difference":       round(diff, 2),
        "pct_difference":   round(pct_diff, 2),
        "t_statistic":      round(stat, 4),
        "p_value":          round(p_value, 6),
        "alpha":            alpha,
        "decision":         decision,
        "significant":      p_value < alpha,
    }


def test_margin(
    group_a: pd.Series,
    group_b: pd.Series,
    group_a_name: str = "Group A",
    group_b_name: str = "Group B",
    alpha: float = 0.05
) -> Dict:
    """
    Z-test for difference in margin (profit) between two groups.

    Margin = TotalPremium - TotalClaims per policy.
    Tests whether two groups have significantly different profitability.

    Args:
        group_a       : Margin series for control group
        group_b       : Margin series for test group
        group_a_name  : label for group A
        group_b_name  : label for group B
        alpha         : significance level (default 0.05)

    Returns:
        dict with test results, p-value, and decision
    """
    stat, p_value = stats.ttest_ind(
        group_a.dropna(),
        group_b.dropna(),
        equal_var=False,
        alternative='two-sided'
    )

    mean_a = group_a.mean()
    mean_b = group_b.mean()
    diff = mean_b - mean_a

    decision = "REJECT H₀" if p_value < alpha else "FAIL TO REJECT H₀"

    return {
        "test_type":      "Welch's t-test (z-test equivalent at large n)",
        "kpi":            "Margin (R)",
        "group_a":        group_a_name,
        "group_b":        group_b_name,
        "n_a":            len(group_a.dropna()),
        "n_b":            len(group_b.dropna()),
        "mean_margin_a":  round(mean_a, 2),
        "mean_margin_b":  round(mean_b, 2),
        "difference":     round(diff, 2),
        "t_statistic":    round(stat, 4),
        "p_value":        round(p_value, 6),
        "alpha":          alpha,
        "decision":       decision,
        "significant":    p_value < alpha,
    }


def test_chi_squared(
    group_a: pd.Series,
    group_b: pd.Series,
    group_a_name: str = "Group A",
    group_b_name: str = "Group B",
    alpha: float = 0.05
) -> Dict:
    """
    Chi-squared test for difference in claim frequency
    between two categorical groups.

    Builds a 2x2 contingency table:
    rows = Group A / Group B
    cols = Claimed / Did not claim

    Args:
        group_a       : TotalClaims series for group A
        group_b       : TotalClaims series for group B
        group_a_name  : label for group A
        group_b_name  : label for group B
        alpha         : significance level (default 0.05)

    Returns:
        dict with test results, p-value, and decision
    """
    # Build contingency table
    # [[claimed_A, not_claimed_A], [claimed_B, not_claimed_B]]
    claimed_a     = (group_a > 0).sum()
    not_claimed_a = (group_a == 0).sum()
    claimed_b     = (group_b > 0).sum()
    not_claimed_b = (group_b == 0).sum()

    contingency_table = np.array([
        [claimed_a, not_claimed_a],
        [claimed_b, not_claimed_b]
    ])

    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

    decision = "REJECT H₀" if p_value < alpha else "FAIL TO REJECT H₀"

    return {
        "test_type":         "Chi-squared Test",
        "kpi":               "Claim Frequency (proportions)",
        "group_a":           group_a_name,
        "group_b":           group_b_name,
        "n_a":               len(group_a),
        "n_b":               len(group_b),
        "claimed_a":         int(claimed_a),
        "claimed_b":         int(claimed_b),
        "freq_a":            round(claimed_a / len(group_a), 6),
        "freq_b":            round(claimed_b / len(group_b), 6),
        "chi2_statistic":    round(chi2, 4),
        "degrees_of_freedom": int(dof),
        "p_value":           round(p_value, 6),
        "alpha":             alpha,
        "decision":          decision,
        "significant":       p_value < alpha,
    }


def print_results(results: Dict) -> None:
    """
    Pretty-print test results to the notebook output.
    Makes results readable without needing a DataFrame.
    """
    import sys
    
    def safe_print(text: str) -> None:
        try:
            print(text)
        except UnicodeEncodeError:
            # Fallback to replacing emojis with ASCII
            text_fixed = (
                text.replace("✅ ", "")
                    .replace("❌ ", "")
                    .replace("⚠️ ", "Warning: ")
            )
            # Encode with replace to handle any other characters
            encoding = sys.stdout.encoding or 'utf-8'
            try:
                print(text_fixed.encode(encoding, errors='replace').decode(encoding))
            except Exception:
                # Absolute fallback
                print(text_fixed.encode('ascii', errors='replace').decode('ascii'))

    if "error" in results and "n_a" not in results:
        safe_print("=" * 60)
        safe_print(f"  ERROR: {results['error']}")
        safe_print("=" * 60)
        return

    safe_print("=" * 60)
    safe_print(f"  TEST: {results.get('test_type', 'Unknown')}")
    safe_print(f"  KPI:  {results.get('kpi', 'Unknown')}")
    safe_print("=" * 60)
    
    n_a = results.get('n_a')
    n_b = results.get('n_b')
    n_a_str = f"{n_a:,}" if n_a is not None else "None"
    n_b_str = f"{n_b:,}" if n_b is not None else "None"
    
    safe_print(f"  Group A ({results.get('group_a')}):  n = {n_a_str}")
    safe_print(f"  Group B ({results.get('group_b')}):  n = {n_b_str}")
    safe_print("-" * 60)

    # Print KPI values depending on test type
    if 'freq_a' in results:
        safe_print(f"  Claim Freq A:    {results['freq_a']:.4%}")
        safe_print(f"  Claim Freq B:    {results['freq_b']:.4%}")
        safe_print(f"  Difference:      {results.get('pct_difference', 0):+.2f}%")
    if 'mean_a' in results:
        safe_print(f"  Mean Claim A:    R {results['mean_a']:,.2f}")
        safe_print(f"  Mean Claim B:    R {results['mean_b']:,.2f}")
        safe_print(f"  Difference:      R {results['difference']:+,.2f} ({results['pct_difference']:+.1f}%)")
    if 'mean_margin_a' in results:
        safe_print(f"  Mean Margin A:   R {results['mean_margin_a']:,.2f}")
        safe_print(f"  Mean Margin B:   R {results['mean_margin_b']:,.2f}")
        safe_print(f"  Difference:      R {results['difference']:+,.2f}")

    safe_print("-" * 60)
    
    p_val = results.get('p_value')
    p_val_str = f"{p_val:.6f}" if p_val is not None else "None"

    safe_print(f"  p-value:         {p_val_str}")
    safe_print(f"  Threshold:       {results.get('alpha')}")
    sig = "✅ YES" if results.get('significant') else "❌ NO"
    safe_print(f"  Significant:     {sig}")
    safe_print(f"  Decision:        {results.get('decision')}")
    if "error" in results:
        safe_print(f"  Note:            ⚠️ {results['error']}")
    safe_print("=" * 60)