"""V2: Categories retriever - Dense + category filtering."""

import sys
from typing import Dict, List, Set

sys.path.insert(0, "liquid-shared-core")

from liquid_shared import VectorStore, RetrievalResult, EmbeddingService
from .base import BaseRetriever, RetrieverConfig


class CategoriesRetriever(BaseRetriever):
    """V2: +Categories - Dense retrieval with category-based boosting."""

    # Category keyword mappings for document classification
    CATEGORY_KEYWORDS = {
        "AI/ML Security": {
            "machine learning", "ml", "ai", "adversarial", "attack",
            "poison", "robustness", "model", "algorithm", "neural"
        },
        "Data Protection": {
            "gdpr", "privacy", "data protection", "personal data",
            "privacy rights", "data security", "compliance"
        },
        "Cyber Insurance": {
            "insurance", "cyber insurance", "coverage", "actuarial",
            "risk assessment", "premium", "underwriting"
        },
        "Technical Controls": {
            "input validation", "monitoring", "anomaly detection",
            "ensemble", "security controls", "defense", "mitigation"
        },
        "Threat Intelligence": {
            "threat", "intelligence", "threat landscape", "attack patterns",
            "indicators", "vulnerability"
        },
        "Risk Management": {
            "incident response", "security risk", "risk management",
            "cybersecurity", "risk assessment"
        },
        "Governance & Policy": {
            "audit", "compliance", "governance", "regulation",
            "policy", "framework", "standards"
        },
        "Research & Innovation": {
            "research", "innovation", "methodology", "analysis",
            "survey", "review"
        }
    }

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
        top_k: int = 10,
        category_boost: float = 0.2,
    ):
        config = RetrieverConfig(
            name="V2: +Categories",
            description="Dense + category filtering",
            top_k=top_k,
        )
        super().__init__(vector_store, embedding_service, config)
        self.category_boost = category_boost  # Score boost for category matches

    def _predict_category(self, query: str) -> str:
        """Predict query category using keyword matching.

        Returns the category with the most keyword matches in the query.
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())

        # Count matches for each category
        category_scores = {}
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            # Check both exact matches and partial matches
            exact_matches = len(query_words & keywords)
            partial_matches = sum(
                1 for word in query_words
                for keyword in keywords
                if keyword in word or word in keyword
            )
            category_scores[category] = exact_matches * 2 + partial_matches

        # Return category with highest score, or "Research & Innovation" if no matches
        if max(category_scores.values()) == 0:
            return "Research & Innovation"

        return max(category_scores.items(), key=lambda x: x[1])[0]

    def _infer_document_category(self, metadata: dict) -> Set[str]:
        """Infer document categories from tags and entities.

        Returns set of categories that match the document's content.
        """
        # Extract tags and entities
        tags = metadata.get('tags', '')
        entities = metadata.get('entities', '')

        doc_text = f"{tags} {entities}".lower()
        doc_words = set(doc_text.split())

        # Find matching categories
        matching_categories = set()
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            # Check for keyword matches
            matches = sum(
                1 for keyword in keywords
                if keyword in doc_text
            )
            # If significant overlap, add category
            if matches >= 2:  # At least 2 keyword matches
                matching_categories.add(category)

        return matching_categories if matching_categories else {"Research & Innovation"}

    def retrieve(self, query: str) -> list[RetrievalResult]:
        """Dense search with category-based boosting.

        Process:
        1. Predict query category
        2. Dense retrieval (larger candidate set)
        3. Infer document categories from tags/entities
        4. Boost scores for category matches
        5. Re-rank and return top_k
        """
        # Step 1: Predict query category
        predicted_category = self._predict_category(query)

        # Step 2: Dense retrieval with larger candidate set
        query_embedding = self._query_to_embedding(query)
        candidate_k = min(self.config.top_k * 2, 50)

        results_dict = self.vector_store.query(
            query_embedding=query_embedding,
            top_k=candidate_k
        )
        results = self._results_to_retrieval_results(results_dict)

        if not results:
            return []

        # Step 3 & 4: Infer categories and boost matching documents
        for result in results:
            doc_categories = self._infer_document_category(result.metadata)

            # Boost score if document matches predicted category
            if predicted_category in doc_categories:
                result.score = min(result.score + self.category_boost, 1.0)

        # Step 5: Re-rank by boosted scores
        reranked = sorted(results, key=lambda r: r.score, reverse=True)

        return reranked[:self.config.top_k]
