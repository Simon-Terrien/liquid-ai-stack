"""Statistical analysis for ablation study."""

import numpy as np
from dataclasses import dataclass
from scipy import stats
from typing import Dict, List, Tuple


@dataclass
class StatisticalComparison:
    """Results of statistical comparison between two variants."""

    variant_a: str
    variant_b: str
    metric: str

    # Mean values
    mean_a: float
    mean_b: float
    mean_diff: float
    percent_improvement: float

    # Statistical tests
    t_statistic: float
    p_value: float
    significant: bool

    # Effect size
    cohens_d: float
    effect_size_interpretation: str

    # Confidence interval
    ci_lower: float
    ci_upper: float
    confidence_level: float


class StatisticalAnalyzer:
    """Statistical analyzer for ablation study results."""

    def __init__(self, significance_level: float = 0.05, confidence_level: float = 0.95):
        """Initialize analyzer.

        Args:
            significance_level: Significance threshold for p-values (default: 0.05)
            confidence_level: Confidence level for intervals (default: 0.95)
        """
        self.significance_level = significance_level
        self.confidence_level = confidence_level

    def compute_cohens_d(self, group_a: np.ndarray, group_b: np.ndarray) -> float:
        """Compute Cohen's d effect size.

        Cohen's d = (mean_a - mean_b) / pooled_std

        Interpretation:
        - Small effect: d = 0.2
        - Medium effect: d = 0.5
        - Large effect: d = 0.8

        Args:
            group_a: Values for group A
            group_b: Values for group B

        Returns:
            Cohen's d effect size
        """
        mean_a = np.mean(group_a)
        mean_b = np.mean(group_b)

        # Pooled standard deviation
        n_a = len(group_a)
        n_b = len(group_b)
        var_a = np.var(group_a, ddof=1)
        var_b = np.var(group_b, ddof=1)
        pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))

        if pooled_std == 0:
            return 0.0

        return (mean_a - mean_b) / pooled_std

    def interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size.

        Args:
            d: Cohen's d value

        Returns:
            Interpretation string
        """
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"

    def paired_t_test(
        self,
        group_a: np.ndarray,
        group_b: np.ndarray,
    ) -> Tuple[float, float]:
        """Perform paired t-test.

        Tests whether the means of two related samples differ significantly.

        Args:
            group_a: Values for group A
            group_b: Values for group B

        Returns:
            Tuple of (t-statistic, p-value)
        """
        t_stat, p_value = stats.ttest_rel(group_a, group_b)
        return t_stat, p_value

    def confidence_interval(
        self,
        data: np.ndarray,
        confidence: float = None,
    ) -> Tuple[float, float]:
        """Compute confidence interval for mean.

        Args:
            data: Data values
            confidence: Confidence level (default: self.confidence_level)

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        confidence = confidence or self.confidence_level
        n = len(data)
        mean = np.mean(data)
        se = stats.sem(data)

        # t-distribution critical value
        ci = se * stats.t.ppf((1 + confidence) / 2, n - 1)

        return mean - ci, mean + ci

    def compare_variants(
        self,
        variant_a_name: str,
        variant_b_name: str,
        metric_name: str,
        values_a: List[float],
        values_b: List[float],
    ) -> StatisticalComparison:
        """Compare two variants on a single metric.

        Args:
            variant_a_name: Name of variant A
            variant_b_name: Name of variant B
            metric_name: Name of the metric being compared
            values_a: Per-query metric values for variant A
            values_b: Per-query metric values for variant B

        Returns:
            Statistical comparison results
        """
        # Convert to numpy arrays
        arr_a = np.array(values_a)
        arr_b = np.array(values_b)

        # Compute means
        mean_a = np.mean(arr_a)
        mean_b = np.mean(arr_b)
        mean_diff = mean_a - mean_b

        # Percent improvement
        if mean_b != 0:
            percent_improvement = (mean_diff / mean_b) * 100
        else:
            percent_improvement = 0.0

        # Paired t-test
        t_stat, p_value = self.paired_t_test(arr_a, arr_b)
        significant = p_value < self.significance_level

        # Cohen's d effect size
        cohens_d = self.compute_cohens_d(arr_a, arr_b)
        effect_interpretation = self.interpret_cohens_d(cohens_d)

        # Confidence interval for difference
        diff = arr_a - arr_b
        ci_lower, ci_upper = self.confidence_interval(diff, self.confidence_level)

        return StatisticalComparison(
            variant_a=variant_a_name,
            variant_b=variant_b_name,
            metric=metric_name,
            mean_a=mean_a,
            mean_b=mean_b,
            mean_diff=mean_diff,
            percent_improvement=percent_improvement,
            t_statistic=t_stat,
            p_value=p_value,
            significant=significant,
            cohens_d=cohens_d,
            effect_size_interpretation=effect_interpretation,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_level=self.confidence_level,
        )

    def compare_to_baseline(
        self,
        baseline_name: str,
        variant_name: str,
        baseline_metrics: Dict[str, List[float]],
        variant_metrics: Dict[str, List[float]],
    ) -> Dict[str, StatisticalComparison]:
        """Compare a variant to baseline across all metrics.

        Args:
            baseline_name: Name of baseline variant
            variant_name: Name of variant being compared
            baseline_metrics: Dict mapping metric_name to per-query values for baseline
            variant_metrics: Dict mapping metric_name to per-query values for variant

        Returns:
            Dict mapping metric_name to statistical comparison
        """
        comparisons = {}

        for metric_name in baseline_metrics.keys():
            if metric_name not in variant_metrics:
                continue

            comparison = self.compare_variants(
                variant_a_name=variant_name,
                variant_b_name=baseline_name,
                metric_name=metric_name,
                values_a=variant_metrics[metric_name],
                values_b=baseline_metrics[metric_name],
            )
            comparisons[metric_name] = comparison

        return comparisons

    def generate_summary_report(
        self,
        comparisons: Dict[str, StatisticalComparison],
    ) -> str:
        """Generate a text summary of statistical comparisons.

        Args:
            comparisons: Dict mapping metric_name to comparison results

        Returns:
            Formatted summary report
        """
        lines = []
        lines.append("=" * 80)
        lines.append("STATISTICAL COMPARISON SUMMARY")
        lines.append("=" * 80)

        for metric_name, comp in comparisons.items():
            lines.append(f"\n{metric_name.upper()}")
            lines.append("-" * 80)
            lines.append(f"  {comp.variant_a}: {comp.mean_a:.4f}")
            lines.append(f"  {comp.variant_b}: {comp.mean_b:.4f}")
            lines.append(f"  Difference: {comp.mean_diff:+.4f} ({comp.percent_improvement:+.2f}%)")
            lines.append(f"  95% CI: [{comp.ci_lower:.4f}, {comp.ci_upper:.4f}]")
            lines.append(f"  t-statistic: {comp.t_statistic:.4f}")
            lines.append(f"  p-value: {comp.p_value:.4f}")
            lines.append(f"  Significant: {'✓ YES' if comp.significant else '✗ NO'} (α={self.significance_level})")
            lines.append(f"  Cohen's d: {comp.cohens_d:.4f} ({comp.effect_size_interpretation})")

        lines.append("\n" + "=" * 80)
        return "\n".join(lines)

    def generate_markdown_table(
        self,
        all_comparisons: Dict[str, Dict[str, StatisticalComparison]],
        metric: str = "recall@5",
    ) -> str:
        """Generate a markdown table comparing all variants on a single metric.

        Args:
            all_comparisons: Dict mapping variant_name to comparison results
            metric: Metric to show in table

        Returns:
            Markdown table
        """
        lines = []
        lines.append(f"| Variant | Mean {metric} | vs Baseline | p-value | Cohen's d | Effect |")
        lines.append("|---------|---------------|-------------|---------|-----------|--------|")

        for variant_name, comparisons in all_comparisons.items():
            if metric not in comparisons:
                continue

            comp = comparisons[metric]
            sig_marker = "*" if comp.significant else ""

            lines.append(
                f"| {variant_name} | {comp.mean_a:.4f} | "
                f"{comp.percent_improvement:+.2f}% | "
                f"{comp.p_value:.4f}{sig_marker} | "
                f"{comp.cohens_d:.4f} | "
                f"{comp.effect_size_interpretation} |"
            )

        lines.append("")
        lines.append("*p < 0.05 indicates significant difference")

        return "\n".join(lines)
