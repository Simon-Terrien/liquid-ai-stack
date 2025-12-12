#!/usr/bin/env python3
"""Main runner for metadata ablation study."""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "liquid-shared-core")
sys.path.insert(0, "experiments")

from liquid_shared import VectorStore, EmbeddingService
from metadata_ablation.config import AblationConfig
from metadata_ablation.variants import create_retriever
from metadata_ablation.evaluator import MetricsEvaluator
from metadata_ablation.statistical import StatisticalAnalyzer


def load_test_queries(queries_path: Path) -> list[dict]:
    """Load test queries from JSON file."""
    with open(queries_path) as f:
        data = json.load(f)
        return data["queries"]


def load_relevance_judgments(judgments_path: Path) -> dict:
    """Load relevance judgments from JSON file."""
    if not judgments_path.exists():
        raise FileNotFoundError(
            f"Relevance judgments not found at {judgments_path}. "
            "Run relevance.py first to generate judgments."
        )

    with open(judgments_path) as f:
        return json.load(f)


def run_variant_experiment(
    variant_id: str,
    vector_store: VectorStore,
    embedding_service: EmbeddingService,
    test_queries: list[dict],
    config: AblationConfig,
) -> dict:
    """Run experiment for a single variant.

    Args:
        variant_id: Variant identifier (v0, v1, etc.)
        vector_store: Vector store with indexed documents
        embedding_service: Service for generating embeddings
        test_queries: List of test queries
        config: Ablation configuration

    Returns:
        Dictionary with results
    """
    print(f"\n{'='*80}")
    print(f"Running {variant_id.upper()}")
    print(f"{'='*80}")

    # Create retriever
    retriever = create_retriever(variant_id, vector_store, embedding_service, top_k=config.top_k)
    print(f"Retriever: {retriever.config.name}")
    print(f"Description: {retriever.config.description}")

    # Run retrieval for each query
    results_by_query = {}
    latencies_by_query = {}

    for i, query in enumerate(test_queries):
        query_id = query["id"]
        query_text = query["text"]

        print(f"\n[{i+1}/{len(test_queries)}] Query {query_id}: {query_text}")

        # Time retrieval
        start_time = time.time()
        results = retriever.retrieve(query_text)
        end_time = time.time()

        latency_ms = (end_time - start_time) * 1000
        latencies_by_query[query_id] = latency_ms

        # Store retrieved doc IDs
        retrieved_doc_ids = [r.chunk_id for r in results]
        results_by_query[query_id] = retrieved_doc_ids

        print(f"  Retrieved: {len(results)} docs in {latency_ms:.2f}ms")
        print(f"  Top 3: {retrieved_doc_ids[:3]}")

    return {
        "variant": variant_id,
        "retriever_config": retriever.get_metadata(),
        "results_by_query": results_by_query,
        "latencies_by_query": latencies_by_query,
    }


def evaluate_variant(
    variant_results: dict,
    relevance_judgments: dict,
    evaluator: MetricsEvaluator,
) -> dict:
    """Evaluate variant results against relevance judgments.

    Args:
        variant_results: Results from run_variant_experiment
        relevance_judgments: Ground truth judgments
        evaluator: Metrics evaluator

    Returns:
        Dictionary with evaluation metrics
    """
    variant_id = variant_results["variant"]
    results_by_query = variant_results["results_by_query"]
    latencies_by_query = variant_results["latencies_by_query"]

    print(f"\n{'='*80}")
    print(f"Evaluating {variant_id.upper()}")
    print(f"{'='*80}")

    # Convert relevance judgments to dict format expected by evaluator
    judgments_by_query = {}
    for query_id in results_by_query.keys():
        if query_id not in relevance_judgments:
            print(f"⚠️  No judgments for query {query_id}, skipping")
            continue

        query_data = relevance_judgments[query_id]
        judgments = query_data["judgments"]

        # Create dict mapping doc_id to relevance score
        judgments_by_query[query_id] = {
            j["doc_id"]: j["relevance"] for j in judgments
        }

    # Evaluate
    metrics = evaluator.evaluate_variant(
        results_by_query=results_by_query,
        judgments_by_query=judgments_by_query,
        latencies_by_query=latencies_by_query,
    )

    # Print summary
    print(f"\nMetrics for {variant_id}:")
    print(f"  Recall@1:     {metrics.recall_at_1:.4f}")
    print(f"  Recall@3:     {metrics.recall_at_3:.4f}")
    print(f"  Recall@5:     {metrics.recall_at_5:.4f}")
    print(f"  Recall@10:    {metrics.recall_at_10:.4f}")
    print(f"  Precision@1:  {metrics.precision_at_1:.4f}")
    print(f"  Precision@3:  {metrics.precision_at_3:.4f}")
    print(f"  Precision@5:  {metrics.precision_at_5:.4f}")
    print(f"  Precision@10: {metrics.precision_at_10:.4f}")
    print(f"  MRR:          {metrics.mrr:.4f}")
    print(f"  NDCG@5:       {metrics.ndcg_at_5:.4f}")
    print(f"  NDCG@10:      {metrics.ndcg_at_10:.4f}")
    print(f"  Avg Latency:  {metrics.avg_latency_ms:.2f}ms")

    return {
        "variant": variant_id,
        "metrics": metrics.to_dict(),
        "results": variant_results,
    }


def run_ablation_study(
    variants: list[str],
    config: AblationConfig,
) -> dict:
    """Run complete ablation study.

    Args:
        variants: List of variant IDs to test
        config: Ablation configuration

    Returns:
        Dictionary with all results
    """
    print("=" * 80)
    print("METADATA ABLATION STUDY")
    print("=" * 80)

    # Load test queries
    print(f"\nLoading test queries from {config.queries_path}")
    test_queries = load_test_queries(config.queries_path)
    print(f"Loaded {len(test_queries)} test queries")

    # Load relevance judgments
    print(f"\nLoading relevance judgments from {config.judgments_path}")
    relevance_judgments = load_relevance_judgments(config.judgments_path)
    print(f"Loaded judgments for {len(relevance_judgments)} queries")

    # Initialize embedding service and vector store
    print(f"\nInitializing vector store from {config.vectordb_path}")
    embedding_service = EmbeddingService()
    vector_store = VectorStore(
        persist_dir=str(config.vectordb_path),
        collection_name=config.collection_name,
    )
    doc_count = vector_store.collection.count()
    print(f"Vector store has {doc_count} documents")

    # Initialize evaluator
    evaluator = MetricsEvaluator(
        k_values=config.k_values,
        ndcg_k_values=config.ndcg_k_values,
    )

    # Run experiments for each variant
    all_results = {}
    for variant_id in variants:
        # Run retrieval experiment
        variant_results = run_variant_experiment(
            variant_id=variant_id,
            vector_store=vector_store,
            embedding_service=embedding_service,
            test_queries=test_queries,
            config=config,
        )

        # Evaluate results
        evaluation = evaluate_variant(
            variant_results=variant_results,
            relevance_judgments=relevance_judgments,
            evaluator=evaluator,
        )

        all_results[variant_id] = evaluation

        # Save intermediate results
        output_file = config.results_path / f"{variant_id}_results.json"
        with open(output_file, "w") as f:
            json.dump(evaluation, f, indent=2)
        print(f"\n✅ Saved results to {output_file}")

    # Statistical analysis
    if "v0" in all_results:
        print(f"\n{'='*80}")
        print("STATISTICAL ANALYSIS")
        print(f"{'='*80}")

        analyzer = StatisticalAnalyzer(
            significance_level=config.significance_level,
            confidence_level=config.confidence_level,
        )

        baseline_name = "v0"
        comparisons = {}

        for variant_id in variants:
            if variant_id == baseline_name:
                continue

            print(f"\nComparing {variant_id} to {baseline_name}...")

            # TODO: Implement per-query metric collection for statistical testing
            # For now, just note that we need per-query values
            print(f"  (Statistical comparison requires per-query metrics)")

        # Save summary
        summary = {
            "config": {
                "variants": variants,
                "test_queries_count": len(test_queries),
                "top_k": config.top_k,
            },
            "results_by_variant": {
                variant_id: result["metrics"]
                for variant_id, result in all_results.items()
            },
        }

        summary_file = config.results_path / "summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n✅ Saved summary to {summary_file}")

    print(f"\n{'='*80}")
    print("ABLATION STUDY COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {config.results_path}")

    return all_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run metadata ablation study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run baseline only
  python experiments/run_ablation.py --variants v0

  # Run all variants
  python experiments/run_ablation.py --variants all

  # Run specific variants
  python experiments/run_ablation.py --variants v0 v1 v5

  # Custom configuration
  python experiments/run_ablation.py --variants all --top-k 20
        """,
    )

    parser.add_argument(
        "--variants",
        nargs="+",
        required=True,
        help="Variant IDs to test (v0-v5) or 'all'",
    )
    parser.add_argument(
        "--queries",
        default="experiments/data/test_queries.json",
        help="Path to test queries JSON",
    )
    parser.add_argument(
        "--judgments",
        default="experiments/data/relevance_judgments.json",
        help="Path to relevance judgments JSON",
    )
    parser.add_argument(
        "--output",
        default="experiments/data/results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of documents to retrieve",
    )

    args = parser.parse_args()

    # Parse variants
    if "all" in args.variants:
        variants = ["v0", "v1", "v2", "v3", "v4", "v5"]
    else:
        variants = args.variants

    # Validate variants
    valid_variants = {"v0", "v1", "v2", "v3", "v4", "v5"}
    invalid = set(variants) - valid_variants
    if invalid:
        print(f"❌ Invalid variants: {invalid}")
        print(f"Valid variants: {sorted(valid_variants)}")
        sys.exit(1)

    # Create config
    config = AblationConfig(
        queries_path=Path(args.queries),
        judgments_path=Path(args.judgments),
        results_path=Path(args.output),
        top_k=args.top_k,
    )

    # Run study
    try:
        run_ablation_study(variants=variants, config=config)
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(130)


if __name__ == "__main__":
    main()
