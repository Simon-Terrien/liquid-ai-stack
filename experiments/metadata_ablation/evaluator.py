"""Metrics evaluator for ablation study."""

import math
from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class MetricScores:
    """Container for evaluation metric scores."""

    # Recall@K scores
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0

    # Precision@K scores
    precision_at_1: float = 0.0
    precision_at_3: float = 0.0
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0

    # MRR
    mrr: float = 0.0

    # NDCG@K scores
    ndcg_at_5: float = 0.0
    ndcg_at_10: float = 0.0

    # Average latency
    avg_latency_ms: float = 0.0

    # Category accuracy (for variants that use categories)
    category_accuracy: float | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "recall@1": self.recall_at_1,
            "recall@3": self.recall_at_3,
            "recall@5": self.recall_at_5,
            "recall@10": self.recall_at_10,
            "precision@1": self.precision_at_1,
            "precision@3": self.precision_at_3,
            "precision@5": self.precision_at_5,
            "precision@10": self.precision_at_10,
            "mrr": self.mrr,
            "ndcg@5": self.ndcg_at_5,
            "ndcg@10": self.ndcg_at_10,
            "avg_latency_ms": self.avg_latency_ms,
            "category_accuracy": self.category_accuracy,
        }


class MetricsEvaluator:
    """Evaluator for computing retrieval metrics."""

    def __init__(self, k_values: list[int] = None, ndcg_k_values: list[int] = None):
        """Initialize evaluator.

        Args:
            k_values: K values for Recall@K and Precision@K (default: [1, 3, 5, 10])
            ndcg_k_values: K values for NDCG@K (default: [5, 10])
        """
        self.k_values = k_values or [1, 3, 5, 10]
        self.ndcg_k_values = ndcg_k_values or [5, 10]

    def compute_recall_at_k(
        self,
        retrieved_docs: list[str],
        relevant_docs: set[str],
        k: int,
    ) -> float:
        """Compute Recall@K.

        Recall@K = (# relevant docs in top-K) / (total # relevant docs)

        Args:
            retrieved_docs: List of retrieved document IDs (ordered by rank)
            relevant_docs: Set of relevant document IDs
            k: Cutoff rank

        Returns:
            Recall@K score (0.0 to 1.0)
        """
        if not relevant_docs:
            return 0.0

        top_k = retrieved_docs[:k]
        relevant_in_top_k = len(set(top_k) & relevant_docs)
        return relevant_in_top_k / len(relevant_docs)

    def compute_precision_at_k(
        self,
        retrieved_docs: list[str],
        relevant_docs: set[str],
        k: int,
    ) -> float:
        """Compute Precision@K.

        Precision@K = (# relevant docs in top-K) / K

        Args:
            retrieved_docs: List of retrieved document IDs (ordered by rank)
            relevant_docs: Set of relevant document IDs
            k: Cutoff rank

        Returns:
            Precision@K score (0.0 to 1.0)
        """
        top_k = retrieved_docs[:k]
        relevant_in_top_k = len(set(top_k) & relevant_docs)
        return relevant_in_top_k / k

    def compute_mrr(
        self,
        retrieved_docs: list[str],
        relevant_docs: set[str],
    ) -> float:
        """Compute Mean Reciprocal Rank.

        MRR = 1 / (rank of first relevant document)

        Args:
            retrieved_docs: List of retrieved document IDs (ordered by rank)
            relevant_docs: Set of relevant document IDs

        Returns:
            MRR score (0.0 to 1.0)
        """
        for i, doc_id in enumerate(retrieved_docs):
            if doc_id in relevant_docs:
                return 1.0 / (i + 1)  # Rank is 1-indexed
        return 0.0

    def compute_dcg_at_k(
        self,
        retrieved_docs: list[str],
        relevance_scores: dict[str, int],
        k: int,
    ) -> float:
        """Compute Discounted Cumulative Gain at K.

        DCG@K = sum_{i=1}^{K} (rel_i / log2(i + 1))

        Args:
            retrieved_docs: List of retrieved document IDs (ordered by rank)
            relevance_scores: Dict mapping doc_id to relevance (0, 1, or 2)
            k: Cutoff rank

        Returns:
            DCG@K score
        """
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_docs[:k]):
            rel = relevance_scores.get(doc_id, 0)
            dcg += rel / math.log2(i + 2)  # i+2 because i is 0-indexed
        return dcg

    def compute_ndcg_at_k(
        self,
        retrieved_docs: list[str],
        relevance_scores: dict[str, int],
        k: int,
    ) -> float:
        """Compute Normalized Discounted Cumulative Gain at K.

        NDCG@K = DCG@K / IDCG@K

        where IDCG@K is the ideal DCG (best possible ranking).

        Args:
            retrieved_docs: List of retrieved document IDs (ordered by rank)
            relevance_scores: Dict mapping doc_id to relevance (0, 1, or 2)
            k: Cutoff rank

        Returns:
            NDCG@K score (0.0 to 1.0)
        """
        # Compute DCG@K
        dcg = self.compute_dcg_at_k(retrieved_docs, relevance_scores, k)

        # Compute IDCG@K (ideal ranking)
        ideal_ranking = sorted(
            relevance_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        ideal_docs = [doc_id for doc_id, _ in ideal_ranking]
        idcg = self.compute_dcg_at_k(ideal_docs, relevance_scores, k)

        if idcg == 0:
            return 0.0

        return dcg / idcg

    def evaluate_query(
        self,
        retrieved_docs: list[str],
        relevance_judgments: dict[str, int],
    ) -> dict[str, float]:
        """Evaluate retrieval for a single query.

        Args:
            retrieved_docs: List of retrieved document IDs (ordered by rank)
            relevance_judgments: Dict mapping doc_id to relevance (0, 1, or 2)

        Returns:
            Dictionary of metric scores
        """
        # Identify relevant docs (relevance >= 1)
        relevant_docs = {
            doc_id for doc_id, rel in relevance_judgments.items() if rel >= 1
        }

        metrics = {}

        # Recall@K
        for k in self.k_values:
            metrics[f"recall@{k}"] = self.compute_recall_at_k(
                retrieved_docs, relevant_docs, k
            )

        # Precision@K
        for k in self.k_values:
            metrics[f"precision@{k}"] = self.compute_precision_at_k(
                retrieved_docs, relevant_docs, k
            )

        # MRR
        metrics["mrr"] = self.compute_mrr(retrieved_docs, relevant_docs)

        # NDCG@K
        for k in self.ndcg_k_values:
            metrics[f"ndcg@{k}"] = self.compute_ndcg_at_k(
                retrieved_docs, relevance_judgments, k
            )

        return metrics

    def aggregate_metrics(self, query_metrics: list[dict[str, float]]) -> MetricScores:
        """Aggregate metrics across all queries.

        Args:
            query_metrics: List of per-query metric dictionaries

        Returns:
            Aggregated metric scores
        """
        if not query_metrics:
            return MetricScores()

        # Compute averages
        avg_metrics = {}
        for key in query_metrics[0].keys():
            values = [m[key] for m in query_metrics]
            avg_metrics[key] = np.mean(values)

        return MetricScores(
            recall_at_1=avg_metrics.get("recall@1", 0.0),
            recall_at_3=avg_metrics.get("recall@3", 0.0),
            recall_at_5=avg_metrics.get("recall@5", 0.0),
            recall_at_10=avg_metrics.get("recall@10", 0.0),
            precision_at_1=avg_metrics.get("precision@1", 0.0),
            precision_at_3=avg_metrics.get("precision@3", 0.0),
            precision_at_5=avg_metrics.get("precision@5", 0.0),
            precision_at_10=avg_metrics.get("precision@10", 0.0),
            mrr=avg_metrics.get("mrr", 0.0),
            ndcg_at_5=avg_metrics.get("ndcg@5", 0.0),
            ndcg_at_10=avg_metrics.get("ndcg@10", 0.0),
        )

    def evaluate_variant(
        self,
        results_by_query: dict[str, list[str]],
        judgments_by_query: dict[str, dict[str, int]],
        latencies_by_query: dict[str, float] | None = None,
    ) -> MetricScores:
        """Evaluate a retrieval variant across all queries.

        Args:
            results_by_query: Dict mapping query_id to list of retrieved doc IDs
            judgments_by_query: Dict mapping query_id to relevance judgments
            latencies_by_query: Optional dict mapping query_id to latency (ms)

        Returns:
            Aggregated metric scores
        """
        query_metrics = []

        for query_id, retrieved_docs in results_by_query.items():
            if query_id not in judgments_by_query:
                continue

            relevance_judgments = judgments_by_query[query_id]
            metrics = self.evaluate_query(retrieved_docs, relevance_judgments)
            query_metrics.append(metrics)

        # Aggregate
        aggregated = self.aggregate_metrics(query_metrics)

        # Add latency if provided
        if latencies_by_query:
            latencies = list(latencies_by_query.values())
            aggregated.avg_latency_ms = np.mean(latencies)

        return aggregated
