"""V0: Baseline retriever - Dense retrieval only."""

import sys

sys.path.insert(0, "liquid-shared-core")

from liquid_shared import VectorStore, RetrievalResult, EmbeddingService
from .base import BaseRetriever, RetrieverConfig


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
