"""RAG quality evaluation metrics.

Implements standard information retrieval metrics for evaluating RAG systems:
- Recall@K
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG)
- Precision@K
- Mean Average Precision (MAP)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RAGMetrics:
    """Container for RAG evaluation metrics."""

    # Retrieval metrics
    recall_at_k: dict[int, float] = field(default_factory=dict)
    precision_at_k: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    map_score: float = 0.0  # Mean Average Precision
    ndcg_at_k: dict[int, float] = field(default_factory=dict)

    # Per-query statistics
    num_queries: int = 0
    queries_with_relevant: int = 0
    avg_relevant_retrieved: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "recall_at_k": {f"recall@{k}": v for k, v in self.recall_at_k.items()},
            "precision_at_k": {f"precision@{k}": v for k, v in self.precision_at_k.items()},
            "mrr": self.mrr,
            "map": self.map_score,
            "ndcg_at_k": {f"ndcg@{k}": v for k, v in self.ndcg_at_k.items()},
            "num_queries": self.num_queries,
            "queries_with_relevant": self.queries_with_relevant,
            "avg_relevant_retrieved": self.avg_relevant_retrieved,
        }

    def __str__(self) -> str:
        """Format metrics as string."""
        lines = [
            f"RAG Evaluation Metrics (n={self.num_queries} queries)",
            f"  MRR: {self.mrr:.4f}",
            f"  MAP: {self.map_score:.4f}",
        ]

        if self.recall_at_k:
            lines.append("  Recall@K:")
            for k in sorted(self.recall_at_k.keys()):
                lines.append(f"    @{k}: {self.recall_at_k[k]:.4f}")

        if self.ndcg_at_k:
            lines.append("  NDCG@K:")
            for k in sorted(self.ndcg_at_k.keys()):
                lines.append(f"    @{k}: {self.ndcg_at_k[k]:.4f}")

        return "\n".join(lines)


class RAGEvaluator:
    """
    Evaluator for RAG system quality.

    Computes standard information retrieval metrics given queries,
    retrieved results, and relevance judgments.
    """

    def __init__(self, k_values: list[int] | None = None):
        """
        Initialize evaluator.

        Args:
            k_values: List of k values for Recall@K and NDCG@K
                     (default: [1, 3, 5, 10, 20])
        """
        self.k_values = k_values or [1, 3, 5, 10, 20]

    def evaluate(
        self,
        queries: list[str],
        retrieved_results: list[list[str]],
        relevance_judgments: list[set[str]],
    ) -> RAGMetrics:
        """
        Evaluate RAG system performance.

        Args:
            queries: List of query strings
            retrieved_results: List of retrieved document IDs per query (ordered by rank)
            relevance_judgments: List of sets of relevant document IDs per query

        Returns:
            RAGMetrics with computed metrics
        """
        if not (len(queries) == len(retrieved_results) == len(relevance_judgments)):
            raise ValueError("Queries, results, and judgments must have same length")

        metrics = RAGMetrics(num_queries=len(queries))

        # Per-query metrics
        recall_scores = {k: [] for k in self.k_values}
        precision_scores = {k: [] for k in self.k_values}
        ndcg_scores = {k: [] for k in self.k_values}
        reciprocal_ranks = []
        average_precisions = []

        for query_idx, (query, retrieved, relevant) in enumerate(
            zip(queries, retrieved_results, relevance_judgments)
        ):
            if not relevant:
                logger.warning(f"Query {query_idx} has no relevant documents, skipping")
                continue

            metrics.queries_with_relevant += 1

            # Compute MRR
            rr = self._reciprocal_rank(retrieved, relevant)
            reciprocal_ranks.append(rr)

            # Compute MAP component (average precision for this query)
            ap = self._average_precision(retrieved, relevant)
            average_precisions.append(ap)

            # Compute Recall@K and Precision@K for each k
            for k in self.k_values:
                recall = self._recall_at_k(retrieved, relevant, k)
                precision = self._precision_at_k(retrieved, relevant, k)
                recall_scores[k].append(recall)
                precision_scores[k].append(precision)

            # Compute NDCG@K for each k
            for k in self.k_values:
                ndcg = self._ndcg_at_k(retrieved, relevant, k)
                ndcg_scores[k].append(ndcg)

        # Aggregate metrics
        metrics.mrr = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
        metrics.map_score = float(np.mean(average_precisions)) if average_precisions else 0.0

        for k in self.k_values:
            metrics.recall_at_k[k] = float(np.mean(recall_scores[k])) if recall_scores[k] else 0.0
            metrics.precision_at_k[k] = (
                float(np.mean(precision_scores[k])) if precision_scores[k] else 0.0
            )
            metrics.ndcg_at_k[k] = float(np.mean(ndcg_scores[k])) if ndcg_scores[k] else 0.0

        # Compute average relevant retrieved
        total_relevant_retrieved = sum(
            len(set(retrieved) & relevant)
            for retrieved, relevant in zip(retrieved_results, relevance_judgments)
            if relevant
        )
        metrics.avg_relevant_retrieved = (
            total_relevant_retrieved / metrics.queries_with_relevant
            if metrics.queries_with_relevant > 0
            else 0.0
        )

        logger.info(f"Evaluated {metrics.num_queries} queries")
        return metrics

    @staticmethod
    def _recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
        """Compute Recall@K."""
        if not relevant:
            return 0.0

        retrieved_at_k = set(retrieved[:k])
        relevant_retrieved = len(retrieved_at_k & relevant)
        return relevant_retrieved / len(relevant)

    @staticmethod
    def _precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
        """Compute Precision@K."""
        if k == 0:
            return 0.0

        retrieved_at_k = retrieved[:k]
        relevant_retrieved = sum(1 for doc_id in retrieved_at_k if doc_id in relevant)
        return relevant_retrieved / min(k, len(retrieved_at_k))

    @staticmethod
    def _reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
        """Compute reciprocal rank (for MRR)."""
        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def _average_precision(retrieved: list[str], relevant: set[str]) -> float:
        """Compute average precision (for MAP)."""
        if not relevant:
            return 0.0

        num_relevant = 0
        sum_precisions = 0.0

        for k, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant:
                num_relevant += 1
                precision_at_k = num_relevant / k
                sum_precisions += precision_at_k

        if num_relevant == 0:
            return 0.0

        return sum_precisions / len(relevant)

    @staticmethod
    def _dcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
        """Compute Discounted Cumulative Gain at K."""
        dcg = 0.0
        for i, doc_id in enumerate(retrieved[:k], start=1):
            # Binary relevance: 1 if relevant, 0 otherwise
            rel = 1.0 if doc_id in relevant else 0.0
            dcg += rel / np.log2(i + 1)
        return dcg

    def _ndcg_at_k(self, retrieved: list[str], relevant: set[str], k: int) -> float:
        """Compute Normalized Discounted Cumulative Gain at K."""
        if not relevant:
            return 0.0

        # Actual DCG
        dcg = self._dcg_at_k(retrieved, relevant, k)

        # Ideal DCG (all relevant docs at top)
        ideal_retrieved = list(relevant) + [doc for doc in retrieved if doc not in relevant]
        idcg = self._dcg_at_k(ideal_retrieved, relevant, k)

        if idcg == 0:
            return 0.0

        return dcg / idcg

    def evaluate_with_scores(
        self,
        queries: list[str],
        retrieved_results: list[list[tuple[str, float]]],
        relevance_judgments: list[set[str]],
    ) -> RAGMetrics:
        """
        Evaluate RAG system with relevance scores.

        Args:
            queries: List of query strings
            retrieved_results: List of (doc_id, score) tuples per query
            relevance_judgments: List of sets of relevant document IDs per query

        Returns:
            RAGMetrics with computed metrics
        """
        # Extract just doc IDs from scored results
        retrieved_ids = [[doc_id for doc_id, _ in results] for results in retrieved_results]

        return self.evaluate(queries, retrieved_ids, relevance_judgments)


def evaluate_rag_from_dict(
    evaluation_data: dict[str, Any],
    k_values: list[int] | None = None,
) -> RAGMetrics:
    """
    Convenience function to evaluate RAG from dictionary format.

    Expected format:
    {
        "queries": ["query1", "query2", ...],
        "retrieved": [["doc1", "doc2", ...], ...],
        "relevant": [["doc1", "doc3", ...], ...]
    }

    Args:
        evaluation_data: Dictionary with queries, retrieved results, and relevance
        k_values: List of k values for metrics

    Returns:
        RAGMetrics with computed metrics
    """
    evaluator = RAGEvaluator(k_values=k_values)

    queries = evaluation_data["queries"]
    retrieved = evaluation_data["retrieved"]
    relevant = [set(r) for r in evaluation_data["relevant"]]

    return evaluator.evaluate(queries, retrieved, relevant)
