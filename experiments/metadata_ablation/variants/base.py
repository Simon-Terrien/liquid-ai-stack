"""Base retriever classes and configuration."""

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
