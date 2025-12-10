"""Statistical testing utilities for experiment validation.

Implements bootstrap testing, confidence intervals, and significance tests
for comparing experimental results.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class SignificanceTestResult:
    """Results of a statistical significance test."""

    statistic: float
    p_value: float
    is_significant: bool
    test_name: str
    effect_size: float | None = None
    confidence_interval: tuple[float, float] | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            "statistic": float(self.statistic),
            "p_value": float(self.p_value),
            "is_significant": self.is_significant,
            "test_name": self.test_name,
            "effect_size": float(self.effect_size) if self.effect_size else None,
            "confidence_interval": (
                [float(self.confidence_interval[0]), float(self.confidence_interval[1])]
                if self.confidence_interval
                else None
            ),
        }


class StatisticalTests:
    """Collection of statistical tests for experiment validation."""

    @staticmethod
    def paired_t_test(
        baseline: Sequence[float],
        treatment: Sequence[float],
        alpha: float = 0.05,
    ) -> SignificanceTestResult:
        """
        Paired t-test for comparing two related samples.

        Use when comparing the same items under two conditions
        (e.g., same documents with different methods).

        Args:
            baseline: Baseline measurements
            treatment: Treatment measurements
            alpha: Significance level (default: 0.05)

        Returns:
            Test result with p-value and significance
        """
        if len(baseline) != len(treatment):
            raise ValueError("Baseline and treatment must have same length for paired test")

        statistic, p_value = stats.ttest_rel(baseline, treatment)

        # Cohen's d for effect size
        diff = np.array(treatment) - np.array(baseline)
        effect_size = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff) > 0 else 0.0

        return SignificanceTestResult(
            statistic=float(statistic),
            p_value=float(p_value),
            is_significant=p_value < alpha,
            test_name="paired_t_test",
            effect_size=float(effect_size),
        )

    @staticmethod
    def independent_t_test(
        baseline: Sequence[float],
        treatment: Sequence[float],
        alpha: float = 0.05,
    ) -> SignificanceTestResult:
        """
        Independent t-test for comparing two independent samples.

        Use when comparing different items under two conditions.

        Args:
            baseline: Baseline measurements
            treatment: Treatment measurements
            alpha: Significance level

        Returns:
            Test result with p-value and significance
        """
        statistic, p_value = stats.ttest_ind(baseline, treatment)

        # Cohen's d for effect size
        pooled_std = np.sqrt(
            ((len(baseline) - 1) * np.var(baseline, ddof=1)
             + (len(treatment) - 1) * np.var(treatment, ddof=1))
            / (len(baseline) + len(treatment) - 2)
        )
        effect_size = (
            (np.mean(treatment) - np.mean(baseline)) / pooled_std
            if pooled_std > 0
            else 0.0
        )

        return SignificanceTestResult(
            statistic=float(statistic),
            p_value=float(p_value),
            is_significant=p_value < alpha,
            test_name="independent_t_test",
            effect_size=float(effect_size),
        )

    @staticmethod
    def wilcoxon_signed_rank(
        baseline: Sequence[float],
        treatment: Sequence[float],
        alpha: float = 0.05,
    ) -> SignificanceTestResult:
        """
        Wilcoxon signed-rank test (non-parametric paired test).

        Use when data is not normally distributed.

        Args:
            baseline: Baseline measurements
            treatment: Treatment measurements
            alpha: Significance level

        Returns:
            Test result with p-value and significance
        """
        if len(baseline) != len(treatment):
            raise ValueError("Baseline and treatment must have same length")

        try:
            result = stats.wilcoxon(baseline, treatment)
            statistic = result.statistic
            p_value = result.pvalue
        except ValueError as e:
            logger.warning(f"Wilcoxon test failed: {e}")
            return SignificanceTestResult(
                statistic=0.0,
                p_value=1.0,
                is_significant=False,
                test_name="wilcoxon_signed_rank",
            )

        # Rank-biserial correlation for effect size
        n = len(baseline)
        effect_size = 1 - (2 * statistic) / (n * (n + 1) / 2)

        return SignificanceTestResult(
            statistic=float(statistic),
            p_value=float(p_value),
            is_significant=p_value < alpha,
            test_name="wilcoxon_signed_rank",
            effect_size=float(effect_size),
        )

    @staticmethod
    def mann_whitney_u(
        baseline: Sequence[float],
        treatment: Sequence[float],
        alpha: float = 0.05,
    ) -> SignificanceTestResult:
        """
        Mann-Whitney U test (non-parametric independent test).

        Use for independent samples with non-normal distribution.

        Args:
            baseline: Baseline measurements
            treatment: Treatment measurements
            alpha: Significance level

        Returns:
            Test result with p-value and significance
        """
        statistic, p_value = stats.mannwhitneyu(baseline, treatment, alternative="two-sided")

        # Rank-biserial correlation for effect size
        n1, n2 = len(baseline), len(treatment)
        effect_size = 1 - (2 * statistic) / (n1 * n2)

        return SignificanceTestResult(
            statistic=float(statistic),
            p_value=float(p_value),
            is_significant=p_value < alpha,
            test_name="mann_whitney_u",
            effect_size=float(effect_size),
        )

    @staticmethod
    def bootstrap_test(
        baseline: Sequence[float],
        treatment: Sequence[float],
        n_bootstrap: int = 10000,
        alpha: float = 0.05,
        statistic_fn=np.mean,
    ) -> SignificanceTestResult:
        """
        Bootstrap hypothesis test for difference in statistics.

        Args:
            baseline: Baseline measurements
            treatment: Treatment measurements
            n_bootstrap: Number of bootstrap samples
            alpha: Significance level
            statistic_fn: Function to compute statistic (default: mean)

        Returns:
            Test result with confidence interval
        """
        baseline_arr = np.array(baseline)
        treatment_arr = np.array(treatment)

        # Observed difference
        observed_diff = statistic_fn(treatment_arr) - statistic_fn(baseline_arr)

        # Bootstrap sampling
        diffs = []
        for _ in range(n_bootstrap):
            # Resample with replacement
            baseline_sample = np.random.choice(baseline_arr, size=len(baseline_arr), replace=True)
            treatment_sample = np.random.choice(
                treatment_arr, size=len(treatment_arr), replace=True
            )
            diff = statistic_fn(treatment_sample) - statistic_fn(baseline_sample)
            diffs.append(diff)

        diffs = np.array(diffs)

        # Compute p-value (two-tailed)
        p_value = np.mean(np.abs(diffs) >= np.abs(observed_diff))

        # Confidence interval
        ci_lower = np.percentile(diffs, (alpha / 2) * 100)
        ci_upper = np.percentile(diffs, (1 - alpha / 2) * 100)

        return SignificanceTestResult(
            statistic=float(observed_diff),
            p_value=float(p_value),
            is_significant=p_value < alpha,
            test_name="bootstrap_test",
            confidence_interval=(float(ci_lower), float(ci_upper)),
        )


def compute_confidence_interval(
    data: Sequence[float],
    confidence: float = 0.95,
    method: str = "bootstrap",
    n_bootstrap: int = 10000,
) -> tuple[float, float, float]:
    """
    Compute confidence interval for data.

    Args:
        data: Sample data
        confidence: Confidence level (0-1)
        method: "bootstrap" or "normal"
        n_bootstrap: Number of bootstrap samples

    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    data_arr = np.array(data)
    mean = float(np.mean(data_arr))

    if method == "bootstrap":
        # Bootstrap confidence interval
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(data_arr, size=len(data_arr), replace=True)
            bootstrap_means.append(np.mean(sample))

        alpha = 1 - confidence
        lower = float(np.percentile(bootstrap_means, (alpha / 2) * 100))
        upper = float(np.percentile(bootstrap_means, (1 - alpha / 2) * 100))

    elif method == "normal":
        # Normal approximation
        sem = stats.sem(data_arr)
        margin = sem * stats.t.ppf((1 + confidence) / 2, len(data_arr) - 1)
        lower = float(mean - margin)
        upper = float(mean + margin)

    else:
        raise ValueError(f"Unknown method: {method}")

    return mean, lower, upper


def compare_multiple_methods(
    results: dict[str, Sequence[float]],
    alpha: float = 0.05,
    baseline_key: str | None = None,
) -> dict[str, dict]:
    """
    Compare multiple methods with statistical tests.

    Args:
        results: Dictionary mapping method names to result lists
        alpha: Significance level
        baseline_key: Key of baseline method (if None, uses first)

    Returns:
        Dictionary with pairwise comparison results
    """
    if baseline_key is None:
        baseline_key = list(results.keys())[0]

    if baseline_key not in results:
        raise ValueError(f"Baseline key '{baseline_key}' not in results")

    baseline = results[baseline_key]
    comparisons = {}

    for method_name, method_results in results.items():
        if method_name == baseline_key:
            continue

        # Use paired test if same length, otherwise independent
        if len(baseline) == len(method_results):
            test_result = StatisticalTests.paired_t_test(baseline, method_results, alpha)
        else:
            test_result = StatisticalTests.independent_t_test(baseline, method_results, alpha)

        # Also compute bootstrap confidence interval
        mean, ci_lower, ci_upper = compute_confidence_interval(method_results)

        comparisons[method_name] = {
            "mean": mean,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "vs_baseline": test_result.to_dict(),
        }

    return {
        "baseline": baseline_key,
        "comparisons": comparisons,
    }
