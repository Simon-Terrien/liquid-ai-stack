"""Visualization tools for ablation study results."""

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class AblationVisualizer:
    """Visualizer for ablation study results."""

    def __init__(self, output_dir: Path):
        """Initialize visualizer.

        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        sns.set_style("whitegrid")
        sns.set_palette("husl")

    def load_results(self, results_path: Path) -> Dict:
        """Load results from JSON file."""
        with open(results_path) as f:
            return json.load(f)

    def plot_metric_comparison(
        self,
        results_by_variant: Dict[str, Dict[str, float]],
        metrics: List[str] = None,
        filename: str = "metric_comparison.png",
    ):
        """Create grouped bar chart comparing metrics across variants.

        Args:
            results_by_variant: Dict mapping variant_id to metrics dict
            metrics: List of metrics to plot (default: Recall@K and Precision@K)
            filename: Output filename
        """
        if metrics is None:
            metrics = ["recall@1", "recall@3", "recall@5", "recall@10"]

        # Prepare data
        variants = list(results_by_variant.keys())
        n_variants = len(variants)
        n_metrics = len(metrics)

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Bar width and positions
        bar_width = 0.8 / n_variants
        x = np.arange(n_metrics)

        # Plot bars for each variant
        for i, variant_id in enumerate(variants):
            variant_metrics = results_by_variant[variant_id]
            values = [variant_metrics.get(m, 0) for m in metrics]

            offset = (i - n_variants / 2) * bar_width + bar_width / 2
            ax.bar(
                x + offset,
                values,
                bar_width,
                label=variant_id.upper(),
                alpha=0.8,
            )

        # Formatting
        ax.set_xlabel("Metric", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Metric Comparison Across Variants", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("@", "@").upper() for m in metrics])
        ax.legend(title="Variant", loc="upper left")
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3)

        # Save
        output_path = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"✅ Saved plot to {output_path}")

    def plot_ablation_heatmap(
        self,
        results_by_variant: Dict[str, Dict[str, float]],
        baseline_id: str = "v0",
        metrics: List[str] = None,
        filename: str = "ablation_heatmap.png",
    ):
        """Create heatmap showing % improvement over baseline.

        Args:
            results_by_variant: Dict mapping variant_id to metrics dict
            baseline_id: Baseline variant ID
            metrics: List of metrics to show
            filename: Output filename
        """
        if metrics is None:
            metrics = [
                "recall@1", "recall@3", "recall@5", "recall@10",
                "precision@1", "precision@3", "precision@5", "precision@10",
                "mrr", "ndcg@5", "ndcg@10"
            ]

        if baseline_id not in results_by_variant:
            print(f"⚠️  Baseline {baseline_id} not found, using absolute values")
            baseline_metrics = {m: 0 for m in metrics}
        else:
            baseline_metrics = results_by_variant[baseline_id]

        # Compute % improvements
        improvements = []
        variant_names = []

        for variant_id, variant_metrics in results_by_variant.items():
            if variant_id == baseline_id:
                continue

            variant_names.append(variant_id.upper())
            row = []
            for metric in metrics:
                baseline_val = baseline_metrics.get(metric, 0)
                variant_val = variant_metrics.get(metric, 0)

                if baseline_val != 0:
                    improvement = ((variant_val - baseline_val) / baseline_val) * 100
                else:
                    improvement = 0

                row.append(improvement)

            improvements.append(row)

        # Create heatmap
        fig, ax = plt.subplots(figsize=(14, 6))

        sns.heatmap(
            improvements,
            annot=True,
            fmt=".1f",
            cmap="RdYlGn",
            center=0,
            cbar_kws={"label": "% Improvement over Baseline"},
            xticklabels=[m.upper() for m in metrics],
            yticklabels=variant_names,
            ax=ax,
        )

        ax.set_title("Ablation Heatmap: % Improvement Over Baseline", fontsize=14, fontweight="bold")
        ax.set_xlabel("Metric", fontsize=12)
        ax.set_ylabel("Variant", fontsize=12)

        # Save
        output_path = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"✅ Saved heatmap to {output_path}")

    def plot_latency_vs_quality(
        self,
        results_by_variant: Dict[str, Dict[str, float]],
        quality_metric: str = "recall@5",
        filename: str = "latency_vs_quality.png",
    ):
        """Create scatter plot of latency vs quality trade-off.

        Args:
            results_by_variant: Dict mapping variant_id to metrics dict
            quality_metric: Metric to use for quality axis
            filename: Output filename
        """
        # Extract data
        variants = []
        latencies = []
        qualities = []

        for variant_id, metrics in results_by_variant.items():
            variants.append(variant_id.upper())
            latencies.append(metrics.get("avg_latency_ms", 0))
            qualities.append(metrics.get(quality_metric, 0))

        # Create scatter plot
        fig, ax = plt.subplots(figsize=(10, 6))

        colors = plt.cm.viridis(np.linspace(0, 1, len(variants)))

        for i, (variant, lat, qual) in enumerate(zip(variants, latencies, qualities)):
            ax.scatter(lat, qual, s=200, c=[colors[i]], alpha=0.7, edgecolors="black", linewidth=1.5)
            ax.annotate(
                variant,
                (lat, qual),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=10,
                fontweight="bold",
            )

        ax.set_xlabel("Average Latency (ms)", fontsize=12)
        ax.set_ylabel(f"{quality_metric.upper()}", fontsize=12)
        ax.set_title("Latency vs Quality Trade-off", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Save
        output_path = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"✅ Saved scatter plot to {output_path}")

    def generate_all_plots(self, summary_path: Path):
        """Generate all visualization plots from summary results.

        Args:
            summary_path: Path to summary.json file
        """
        print(f"\nGenerating visualizations from {summary_path}")

        # Load results
        summary = self.load_results(summary_path)
        results_by_variant = summary["results_by_variant"]

        print(f"Loaded results for {len(results_by_variant)} variants")

        # Generate plots
        self.plot_metric_comparison(results_by_variant)
        self.plot_ablation_heatmap(results_by_variant)
        self.plot_latency_vs_quality(results_by_variant)

        print(f"\n✅ All visualizations saved to {self.output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate ablation study visualizations")
    parser.add_argument(
        "--results",
        default="experiments/data/results/summary.json",
        help="Path to summary.json",
    )
    parser.add_argument(
        "--output",
        default="experiments/data/results/plots",
        help="Output directory for plots",
    )
    args = parser.parse_args()

    visualizer = AblationVisualizer(output_dir=Path(args.output))
    visualizer.generate_all_plots(summary_path=Path(args.results))
