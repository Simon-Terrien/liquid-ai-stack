#!/usr/bin/env python3
"""
Command-line interface for running research validation experiments.

This script provides a convenient way to run all experiments needed for
publication validation.
"""

import argparse
import json
import logging
from pathlib import Path

from liquid_shared.experiments.orchestrator import run_research_validation


def main():
    parser = argparse.ArgumentParser(
        description="Run research validation experiments for liquid-ai-stack",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all experiments
  python run_experiments.py --all

  # Run specific experiments
  python run_experiments.py --experiments dual_pipeline rag_quality

  # Run with MLflow tracking
  python run_experiments.py --all --mlflow-uri http://localhost:5000

  # Specify output directory
  python run_experiments.py --all --output-dir /path/to/results

Available experiments:
  - dual_pipeline:  Compare unified vs separate ETL pipelines
  - rag_quality:    Evaluate RAG retrieval quality
  - multi_model:    Multi-model ablation study (coming soon)
  - ft_quality:     Fine-tuning data quality assessment (coming soon)
        """,
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all available experiments",
    )

    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=["dual_pipeline", "rag_quality", "multi_model", "ft_quality"],
        help="Specific experiments to run",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/experiments",
        help="Directory for experiment outputs (default: data/experiments)",
    )

    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default=None,
        help="MLflow tracking server URI (e.g., http://localhost:5000)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging",
    )

    parser.add_argument(
        "--summary",
        action="store_true",
        help="Only print summary of results (don't run experiments)",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger = logging.getLogger(__name__)

    # Determine which experiments to run
    if args.summary:
        # Just print existing results
        results_file = Path(args.output_dir) / "all_results.json"
        if not results_file.exists():
            logger.error(f"No results found at {results_file}")
            return 1

        with open(results_file) as f:
            results = json.load(f)

        print("\n" + "=" * 80)
        print("EXPERIMENT RESULTS SUMMARY")
        print("=" * 80)
        print(json.dumps(results, indent=2))
        return 0

    experiments = None
    if not args.all:
        if not args.experiments:
            parser.error("Must specify --all or --experiments")
        experiments = args.experiments
    else:
        experiments = ["dual_pipeline", "rag_quality"]

    logger.info(f"Running experiments: {experiments}")
    logger.info(f"Output directory: {args.output_dir}")

    # Run experiments
    try:
        results = run_research_validation(
            output_dir=args.output_dir,
            experiments=experiments,
            mlflow_tracking_uri=args.mlflow_uri,
        )

        # Print summary
        print("\n" + "=" * 80)
        print("EXPERIMENTS COMPLETE")
        print("=" * 80)
        print(f"\nResults saved to: {args.output_dir}/")
        print(f"Report: {args.output_dir}/experiment_report.md")
        print(f"JSON: {args.output_dir}/all_results.json")

        if args.mlflow_uri:
            print(f"\nMLflow tracking: {args.mlflow_uri}")

        # Print key metrics
        print("\nKey Results:")
        for exp_name, exp_results in results.items():
            if "error" in exp_results:
                print(f"\n  {exp_name}: ERROR - {exp_results['error']}")
                continue

            print(f"\n  {exp_name}:")

            # Dual pipeline metrics
            if exp_name == "dual_pipeline" and "efficiency_gains" in exp_results:
                gains = exp_results["efficiency_gains"]
                print(f"    Time savings: {gains.get('time_saving_percent', 0):.2f}%")
                print(f"    GPU savings: {gains.get('gpu_saving_percent', 0):.2f}%")
                print(f"    Speedup: {gains.get('speedup_factor', 0):.2f}x")

            # RAG quality metrics
            elif exp_name == "rag_quality" and "results" in exp_results:
                for variant, variant_results in exp_results["results"].items():
                    if "metrics" in variant_results:
                        metrics = variant_results["metrics"]
                        recall_5 = metrics.get("recall_at_k", {}).get("recall@5", 0)
                        mrr = metrics.get("mrr", 0)
                        print(f"    {variant}: Recall@5={recall_5:.4f}, MRR={mrr:.4f}")

        print("\n" + "=" * 80)

        return 0

    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
