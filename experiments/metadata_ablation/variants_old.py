"""Retrieval variant implementations for ablation study."""

import sys
from abc import ABC, abstractmethod
from typing import Any

sys.path.insert(0, "liquid-shared-core")

from liquid_shared import VectorStore, RetrievalResult, EmbeddingService
from pydantic import BaseModel


class RetrieverConfig(BaseModel):
    """Configuration for a retrieval variant."""

    name: str
    description: str
    top_k: int = 10
    vector_weight: float = 0.7
    bm25_weight: float = 0.3
    importance_weight: float = 0.2


class BaseRetriever(ABC):
    """Base class for retrieval variants."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
        config: RetrieverConfig,
    ):
        """Initialize retriever.

        Args:
            vector_store: Vector store with indexed documents
            embedding_service: Service for generating embeddings
            config: Retriever configuration
        """
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.config = config

    def _query_to_embedding(self, query: str) -> list[float]:
        """Convert query text to embedding."""
        embeddings = self.embedding_service.encode([query])
        # Handle both numpy arrays and lists
        if hasattr(embeddings[0], 'tolist'):
            return embeddings[0].tolist()
        else:
            return embeddings[0]

    def _results_to_retrieval_results(
        self, query_results: dict[str, Any]
    ) -> list[RetrievalResult]:
        """Convert Chroma results to RetrievalResult objects."""
        results = []
        ids = query_results.get("ids", [[]])[0]
        documents = query_results.get("documents", [[]])[0]
        distances = query_results.get("distances", [[]])[0]
        metadatas = query_results.get("metadatas", [[]])[0]

        for i, chunk_id in enumerate(ids):
            # Convert distance to similarity score (assuming cosine distance)
            # Cosine distance is in [0, 2], convert to similarity in [0, 1]
            distance = distances[i] if i < len(distances) else 1.0
            score = 1 - (distance / 2.0)  # Normalize to [0, 1]

            results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    text=documents[i] if i < len(documents) else "",
                    score=max(0.0, min(1.0, score)),  # Clamp to [0, 1]
                    metadata=metadatas[i] if i < len(metadatas) else {},
                )
            )

        return results

    @abstractmethod
    def retrieve(self, query: str) -> list[RetrievalResult]:
        """Retrieve documents for query.

        Args:
            query: Search query

        Returns:
            List of retrieval results
        """
        pass

    def get_metadata(self) -> dict[str, Any]:
        """Get retriever metadata for logging."""
        return {
            "name": self.config.name,
            "description": self.config.description,
            "config": self.config.model_dump(),
        }


class BaselineRetriever(BaseRetriever):
    """V0: Baseline - Dense retrieval only (no enhanced metadata)."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
        top_k: int = 10,
    ):
        config = RetrieverConfig(
            name="V0: Baseline",
            description="Dense retrieval only (tags + entities)",
            top_k=top_k,
        )
        super().__init__(vector_store, embedding_service, config)

    def retrieve(self, query: str) -> list[RetrievalResult]:
        """Simple vector similarity search."""
        query_embedding = self._query_to_embedding(query)
        results = self.vector_store.query(
            query_embedding=query_embedding, top_k=self.config.top_k
        )
        return self._results_to_retrieval_results(results)


class KeywordsRetriever(BaseRetriever):
    """V1: +Keywords - Dense + BM25 keyword matching."""

    def __init__(self, vector_store: VectorStore, top_k: int = 10, bm25_weight: float = 0.3):
        config = RetrieverConfig(
            name="V1: +Keywords",
            description="Dense + BM25 keyword matching",
            top_k=top_k,
            bm25_weight=bm25_weight,
        )
        super().__init__(vector_store, config)

    def retrieve(self, query: str) -> list[RetrievalResult]:
        """Hybrid search using dense + BM25 on keywords field."""
        # For now, use standard dense search
        # TODO: Implement BM25 hybrid search when ChromaDB supports it
        # This would combine vector similarity with keyword matching on metadata.keywords
        return self.vector_store.search(query, top_k=self.config.top_k)


class CategoriesRetriever(BaseRetriever):
    """V2: +Categories - Dense + category filtering."""

    def __init__(self, vector_store: VectorStore, top_k: int = 10):
        config = RetrieverConfig(
            name="V2: +Categories",
            description="Dense + category filtering",
            top_k=top_k,
        )
        super().__init__(vector_store, config)

    def _predict_category(self, query: str) -> str:
        """Predict query category using simple keyword matching.

        TODO: Replace with LLM-based classification for better accuracy.
        """
        query_lower = query.lower()

        # Simple rule-based classification
        if any(term in query_lower for term in ["adversarial", "attack", "poison", "robustness", "ai", "ml", "model"]):
            return "AI/ML Security"
        elif any(term in query_lower for term in ["gdpr", "privacy", "data protection", "personal data"]):
            return "Data Protection"
        elif any(term in query_lower for term in ["insurance", "coverage", "risk assessment"]):
            return "Cyber Insurance"
        elif any(term in query_lower for term in ["input validation", "monitoring", "anomaly detection", "ensemble"]):
            return "Technical Controls"
        elif any(term in query_lower for term in ["threat intelligence", "intelligence"]):
            return "Threat Intelligence"
        elif any(term in query_lower for term in ["incident response", "security risk"]):
            return "Risk Management"
        elif any(term in query_lower for term in ["audit", "compliance", "governance", "regulation"]):
            return "Governance & Policy"
        else:
            return "Research & Innovation"

    def retrieve(self, query: str) -> list[RetrievalResult]:
        """Dense search with category filtering."""
        # Predict query category
        predicted_category = self._predict_category(query)

        # For now, retrieve all and filter
        # TODO: Add ChromaDB where clause filtering when metadata structure is confirmed
        results = self.vector_store.search(query, top_k=self.config.top_k * 2)

        # Filter by category
        filtered = [
            r for r in results
            if "categories" in r.metadata
            and predicted_category in r.metadata.get("categories", [])
        ]

        # If not enough results, return unfiltered
        if len(filtered) < self.config.top_k // 2:
            return results[:self.config.top_k]

        return filtered[:self.config.top_k]


class TaxonomyRetriever(BaseRetriever):
    """V3: +Taxonomy - Dense + taxonomy-based query expansion."""

    def __init__(self, vector_store: VectorStore, top_k: int = 10):
        config = RetrieverConfig(
            name="V3: +Taxonomy",
            description="Dense + taxonomy expansion",
            top_k=top_k,
        )
        super().__init__(vector_store, config)

    def _expand_query_with_taxonomy(self, query: str) -> list[str]:
        """Expand query with related taxonomy terms.

        TODO: Implement taxonomy-based expansion using graph traversal.
        For now, return simple synonyms.
        """
        query_lower = query.lower()
        expansion_terms = []

        # Simple expansion rules
        if "adversarial" in query_lower:
            expansion_terms.extend(["attack", "perturbation", "robustness"])
        if "gdpr" in query_lower:
            expansion_terms.extend(["data protection", "privacy", "compliance"])
        if "insurance" in query_lower:
            expansion_terms.extend(["risk", "coverage", "cyber insurance"])

        return expansion_terms

    def retrieve(self, query: str) -> list[RetrievalResult]:
        """Dense search with query expansion."""
        # Expand query
        expansion_terms = self._expand_query_with_taxonomy(query)
        expanded_query = f"{query} {' '.join(expansion_terms)}"

        return self.vector_store.search(expanded_query, top_k=self.config.top_k)


class HybridRetriever(BaseRetriever):
    """V4: Keywords+Categories - Combines BM25 keywords with category filtering."""

    def __init__(
        self,
        vector_store: VectorStore,
        top_k: int = 10,
        bm25_weight: float = 0.3,
    ):
        config = RetrieverConfig(
            name="V4: Keywords+Categories",
            description="Hybrid (dense + BM25) + category filtering",
            top_k=top_k,
            bm25_weight=bm25_weight,
        )
        super().__init__(vector_store, config)

    def _predict_category(self, query: str) -> str:
        """Predict query category (same as CategoriesRetriever)."""
        # Reuse category prediction logic
        return CategoriesRetriever(self.vector_store)._predict_category(query)

    def retrieve(self, query: str) -> list[RetrievalResult]:
        """Hybrid search with category filtering."""
        # Predict category
        predicted_category = self._predict_category(query)

        # Hybrid search (for now, just dense)
        # TODO: Implement BM25 hybrid when available
        results = self.vector_store.search(query, top_k=self.config.top_k * 2)

        # Filter by category
        filtered = [
            r for r in results
            if "categories" in r.metadata
            and predicted_category in r.metadata.get("categories", [])
        ]

        # If not enough results, return unfiltered
        if len(filtered) < self.config.top_k // 2:
            return results[:self.config.top_k]

        return filtered[:self.config.top_k]


class FullEnhancedRetriever(BaseRetriever):
    """V5: Full Enhanced - All metadata features with importance weighting."""

    def __init__(
        self,
        vector_store: VectorStore,
        top_k: int = 10,
        bm25_weight: float = 0.3,
        importance_weight: float = 0.2,
    ):
        config = RetrieverConfig(
            name="V5: Full Enhanced",
            description="All features + importance weighting",
            top_k=top_k,
            bm25_weight=bm25_weight,
            importance_weight=importance_weight,
        )
        super().__init__(vector_store, config)

    def _predict_category(self, query: str) -> str:
        """Predict query category."""
        return CategoriesRetriever(self.vector_store)._predict_category(query)

    def _expand_query_with_taxonomy(self, query: str) -> list[str]:
        """Expand query with taxonomy terms."""
        return TaxonomyRetriever(self.vector_store)._expand_query_with_taxonomy(query)

    def _rerank_by_importance(self, results: list[RetrievalResult]) -> list[RetrievalResult]:
        """Rerank results by combining similarity score with importance.

        Final score = similarity * (1 - importance_weight) + (importance / 10) * importance_weight
        """
        for result in results:
            importance = result.metadata.get("importance", 5)
            normalized_importance = importance / 10.0

            # Combine scores
            original_score = result.score
            result.score = (
                original_score * (1 - self.config.importance_weight)
                + normalized_importance * self.config.importance_weight
            )

        # Re-sort by new scores
        return sorted(results, key=lambda r: r.score, reverse=True)

    def retrieve(self, query: str) -> list[RetrievalResult]:
        """Full enhanced retrieval with all features."""
        # 1. Predict category
        predicted_category = self._predict_category(query)

        # 2. Expand query with taxonomy
        expansion_terms = self._expand_query_with_taxonomy(query)
        expanded_query = f"{query} {' '.join(expansion_terms)}"

        # 3. Hybrid search (for now, just dense with expanded query)
        # TODO: Add BM25 when available
        results = self.vector_store.search(expanded_query, top_k=self.config.top_k * 2)

        # 4. Filter by category
        filtered = [
            r for r in results
            if "categories" in r.metadata
            and predicted_category in r.metadata.get("categories", [])
        ]

        # If not enough results, use unfiltered
        if len(filtered) < self.config.top_k // 2:
            filtered = results

        # 5. Rerank by importance
        reranked = self._rerank_by_importance(filtered)

        return reranked[:self.config.top_k]


# Factory function for creating retrievers
def create_retriever(
    variant: str,
    vector_store: VectorStore,
    embedding_service: EmbeddingService,
    top_k: int = 10,
) -> BaseRetriever:
    """Create retriever for specified variant.

    Args:
        variant: Variant ID (v0, v1, v2, v3, v4, v5)
        vector_store: Vector store with indexed documents
        embedding_service: Service for generating embeddings
        top_k: Number of results to retrieve

    Returns:
        Retriever instance

    Raises:
        ValueError: If variant is invalid
    """
    variant = variant.lower()

    if variant == "v0":
        return BaselineRetriever(vector_store, embedding_service, top_k=top_k)
    elif variant == "v1":
        return KeywordsRetriever(vector_store, embedding_service, top_k=top_k)
    elif variant == "v2":
        return CategoriesRetriever(vector_store, embedding_service, top_k=top_k)
    elif variant == "v3":
        return TaxonomyRetriever(vector_store, embedding_service, top_k=top_k)
    elif variant == "v4":
        return HybridRetriever(vector_store, embedding_service, top_k=top_k)
    elif variant == "v5":
        return FullEnhancedRetriever(vector_store, embedding_service, top_k=top_k)
    else:
        raise ValueError(f"Invalid variant: {variant}. Must be v0, v1, v2, v3, v4, or v5")
