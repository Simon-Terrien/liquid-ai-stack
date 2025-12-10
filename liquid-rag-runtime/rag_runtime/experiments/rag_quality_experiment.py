"""RAG quality evaluation experiment.

Evaluates RAG system quality using:
- Recall@K (K = 1, 3, 5, 10)
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG)
- Answer accuracy (using LLM-as-judge or exact match)

Compares different retrieval strategies:
- Dense retrieval only
- BM25 (sparse) only
- Hybrid (dense + BM25)
- Hybrid with metadata boosting
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from liquid_shared.experiments import (
    ExperimentConfig,
    ExperimentTracker,
    ModelConfig,
    DatasetConfig,
    EvalConfig,
    set_seed,
    StatisticalTests,
)
from liquid_shared.experiments.evaluators import RAGEvaluator, RAGMetrics
from liquid_shared import VectorStore, EmbeddingService, DATA_DIR

logger = logging.getLogger(__name__)


@dataclass
class QueryExample:
    """A query with relevance judgments."""

    query: str
    relevant_doc_ids: set[str]
    expected_answer: str | None = None


@dataclass
class RetrievalVariantResults:
    """Results for a retrieval strategy variant."""

    variant_name: str
    metrics: RAGMetrics
    avg_retrieval_time_ms: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "variant_name": self.variant_name,
            "metrics": self.metrics.to_dict(),
            "avg_retrieval_time_ms": self.avg_retrieval_time_ms,
        }


class RAGQualityExperiment:
    """
    Experiment to evaluate RAG system quality.

    Tests different retrieval strategies and measures their effectiveness
    using standard IR metrics.
    """

    def __init__(self, config: ExperimentConfig):
        """
        Initialize experiment.

        Args:
            config: Experiment configuration
        """
        self.config = config
        self.output_dir = config.output_dir / "rag_quality"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup tracker
        self.tracker = ExperimentTracker(
            experiment_name=config.mlflow_experiment_name or "rag-quality-evaluation",
            run_name=config.name,
            tracking_uri=config.mlflow_tracking_uri,
            output_dir=self.output_dir,
        )

        # Initialize vector store and embedding service
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore(
            collection_name="documents",
            embedding_service=self.embedding_service,
            persist_directory=str(DATA_DIR / "vectordb"),
        )

        # Initialize evaluator
        self.evaluator = RAGEvaluator(k_values=config.eval.recall_k_values)

    def run(self, test_queries: list[QueryExample]) -> dict[str, Any]:
        """
        Run RAG quality evaluation experiment.

        Args:
            test_queries: List of queries with relevance judgments

        Returns:
            Dictionary with results and statistical comparisons
        """
        logger.info(f"Starting RAG quality experiment with {len(test_queries)} queries")

        with self.tracker:
            # Log config
            self.tracker.log_params(self.config.to_dict())
            set_seed(self.config.seed)

            # Run evaluation for each retrieval variant
            results = {}

            # 1. Dense retrieval only
            logger.info("Evaluating dense retrieval...")
            results["dense"] = self._evaluate_dense_retrieval(test_queries)
            self._log_variant_metrics("dense", results["dense"])

            # 2. BM25 retrieval (if supported)
            if hasattr(self.vector_store, "search_bm25"):
                logger.info("Evaluating BM25 retrieval...")
                results["bm25"] = self._evaluate_bm25_retrieval(test_queries)
                self._log_variant_metrics("bm25", results["bm25"])

            # 3. Hybrid retrieval
            if hasattr(self.vector_store, "hybrid_search"):
                logger.info("Evaluating hybrid retrieval...")
                results["hybrid"] = self._evaluate_hybrid_retrieval(test_queries)
                self._log_variant_metrics("hybrid", results["hybrid"])

            # 4. Hybrid with metadata boosting
            logger.info("Evaluating hybrid + metadata...")
            results["hybrid_metadata"] = self._evaluate_hybrid_metadata(test_queries)
            self._log_variant_metrics("hybrid_metadata", results["hybrid_metadata"])

            # Compute statistical comparisons
            stat_comparisons = self._compute_statistical_comparisons(results)
            self.tracker.log_dict(stat_comparisons, "statistical_comparisons.json")

            # Save detailed results
            final_results = {
                "config": self.config.to_dict(),
                "num_queries": len(test_queries),
                "results": {k: v.to_dict() for k, v in results.items()},
                "statistical_comparisons": stat_comparisons,
            }

            self.tracker.log_dict(final_results, "rag_quality_results.json")

            logger.info("RAG quality experiment complete")
            return final_results

    def _evaluate_dense_retrieval(
        self, test_queries: list[QueryExample]
    ) -> RetrievalVariantResults:
        """Evaluate dense retrieval strategy."""
        queries = [q.query for q in test_queries]
        relevance_judgments = [q.relevant_doc_ids for q in test_queries]

        retrieved_results = []
        retrieval_times = []

        import time

        for query in queries:
            start = time.time()
            results = self.vector_store.search(
                query,
                top_k=max(self.config.eval.recall_k_values),
            )
            retrieval_times.append((time.time() - start) * 1000)  # ms

            # Extract doc IDs
            doc_ids = [r.chunk_id for r in results]
            retrieved_results.append(doc_ids)

        # Evaluate
        metrics = self.evaluator.evaluate(queries, retrieved_results, relevance_judgments)

        avg_time = sum(retrieval_times) / len(retrieval_times) if retrieval_times else 0

        return RetrievalVariantResults(
            variant_name="dense",
            metrics=metrics,
            avg_retrieval_time_ms=avg_time,
        )

    def _evaluate_bm25_retrieval(
        self, test_queries: list[QueryExample]
    ) -> RetrievalVariantResults:
        """Evaluate BM25 (sparse) retrieval strategy."""
        # TODO: Implement BM25 search in VectorStore
        logger.warning("BM25 retrieval not yet fully implemented")

        # Placeholder: return dummy results
        queries = [q.query for q in test_queries]
        relevance_judgments = [q.relevant_doc_ids for q in test_queries]

        # Use dense retrieval as approximation for now
        retrieved_results = []
        for query in queries:
            results = self.vector_store.search(query, top_k=10)
            doc_ids = [r.chunk_id for r in results]
            retrieved_results.append(doc_ids)

        metrics = self.evaluator.evaluate(queries, retrieved_results, relevance_judgments)

        return RetrievalVariantResults(
            variant_name="bm25",
            metrics=metrics,
            avg_retrieval_time_ms=0.0,
        )

    def _evaluate_hybrid_retrieval(
        self, test_queries: list[QueryExample]
    ) -> RetrievalVariantResults:
        """Evaluate hybrid (dense + BM25) retrieval strategy."""
        # TODO: Implement hybrid search
        logger.warning("Hybrid retrieval not yet fully implemented")

        queries = [q.query for q in test_queries]
        relevance_judgments = [q.relevant_doc_ids for q in test_queries]

        retrieved_results = []
        for query in queries:
            results = self.vector_store.search(query, top_k=10)
            doc_ids = [r.chunk_id for r in results]
            retrieved_results.append(doc_ids)

        metrics = self.evaluator.evaluate(queries, retrieved_results, relevance_judgments)

        return RetrievalVariantResults(
            variant_name="hybrid",
            metrics=metrics,
            avg_retrieval_time_ms=0.0,
        )

    def _evaluate_hybrid_metadata(
        self, test_queries: list[QueryExample]
    ) -> RetrievalVariantResults:
        """Evaluate hybrid retrieval with metadata boosting."""
        # TODO: Implement metadata boosting
        logger.warning("Metadata boosting not yet fully implemented")

        queries = [q.query for q in test_queries]
        relevance_judgments = [q.relevant_doc_ids for q in test_queries]

        retrieved_results = []
        for query in queries:
            # Could boost results based on metadata importance scores
            results = self.vector_store.search(query, top_k=10)
            doc_ids = [r.chunk_id for r in results]
            retrieved_results.append(doc_ids)

        metrics = self.evaluator.evaluate(queries, retrieved_results, relevance_judgments)

        return RetrievalVariantResults(
            variant_name="hybrid_metadata",
            metrics=metrics,
            avg_retrieval_time_ms=0.0,
        )

    def _compute_statistical_comparisons(
        self, results: dict[str, RetrievalVariantResults]
    ) -> dict[str, Any]:
        """Compute statistical significance of differences between variants."""
        comparisons = {}

        if "dense" not in results:
            return comparisons

        baseline = results["dense"]

        for variant_name, variant_results in results.items():
            if variant_name == "dense":
                continue

            # Compare Recall@5 (example)
            if 5 in baseline.metrics.recall_at_k and 5 in variant_results.metrics.recall_at_k:
                # Note: For proper comparison, we need per-query scores
                # This is a simplified version
                baseline_score = baseline.metrics.recall_at_k[5]
                variant_score = variant_results.metrics.recall_at_k[5]

                comparisons[f"{variant_name}_vs_dense"] = {
                    "baseline_recall@5": baseline_score,
                    "variant_recall@5": variant_score,
                    "improvement_percent": (
                        (variant_score - baseline_score) / baseline_score * 100
                        if baseline_score > 0
                        else 0
                    ),
                }

        return comparisons

    def _log_variant_metrics(
        self, variant_name: str, results: RetrievalVariantResults
    ) -> None:
        """Log metrics for a retrieval variant."""
        metrics = results.metrics.to_dict()

        # Flatten metrics for logging
        log_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    log_metrics[f"{variant_name}/{k}"] = v
            else:
                log_metrics[f"{variant_name}/{key}"] = value

        log_metrics[f"{variant_name}/avg_retrieval_time_ms"] = results.avg_retrieval_time_ms

        self.tracker.log_metrics(log_metrics)


def create_test_queries_from_qa_dataset(qa_dataset_path: Path) -> list[QueryExample]:
    """
    Create test queries from a QA dataset generated by ETL.

    Args:
        qa_dataset_path: Path to JSONL file with QA pairs

    Returns:
        List of QueryExample instances
    """
    queries = []

    with open(qa_dataset_path) as f:
        for line in f:
            qa_pair = json.loads(line)

            # Use question as query, assume the chunk it came from is relevant
            queries.append(
                QueryExample(
                    query=qa_pair["instruction"],
                    relevant_doc_ids={qa_pair.get("chunk_id", "unknown")},
                    expected_answer=qa_pair.get("output"),
                )
            )

    return queries


def run_rag_quality_experiment(
    config: ExperimentConfig | None = None,
    test_queries: list[QueryExample] | None = None,
) -> dict[str, Any]:
    """
    Run RAG quality evaluation experiment.

    Args:
        config: Experiment configuration
        test_queries: Test queries with relevance judgments

    Returns:
        Experiment results
    """
    if config is None:
        from liquid_shared.experiments import ModelConfig, DatasetConfig, EvalConfig

        config = ExperimentConfig(
            name="rag_quality_evaluation",
            description="Evaluate RAG system quality with different retrieval strategies",
            experiment_type="rag_quality",
            model=ModelConfig(name="LFM2-700M"),
            dataset=DatasetConfig(name="default"),
            eval=EvalConfig(recall_k_values=[1, 3, 5, 10]),
        )

    if test_queries is None:
        # Try to load from default location
        qa_path = DATA_DIR / "ft" / "qa_pairs.jsonl"
        if qa_path.exists():
            logger.info(f"Loading test queries from {qa_path}")
            test_queries = create_test_queries_from_qa_dataset(qa_path)[:50]  # Sample 50
        else:
            raise ValueError("No test queries provided and default dataset not found")

    experiment = RAGQualityExperiment(config)
    return experiment.run(test_queries)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    results = run_rag_quality_experiment()
    print("\n=== RAG Quality Experiment Results ===")
    for variant_name, variant_results in results["results"].items():
        print(f"\n{variant_name}:")
        print(f"  Recall@5: {variant_results['metrics']['recall_at_k'].get('recall@5', 0):.4f}")
        print(f"  MRR: {variant_results['metrics']['mrr']:.4f}")
        print(f"  NDCG@5: {variant_results['metrics']['ndcg_at_k'].get('ndcg@5', 0):.4f}")
