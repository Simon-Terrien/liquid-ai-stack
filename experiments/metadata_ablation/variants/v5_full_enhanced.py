"""V5: Full Enhanced retriever - All metadata features with importance weighting."""

import sys
import math
from collections import Counter
from typing import List, Dict, Set

sys.path.insert(0, "liquid-shared-core")

from liquid_shared import VectorStore, RetrievalResult, EmbeddingService
from .base import BaseRetriever, RetrieverConfig
from .v1_keywords import KeywordsRetriever
from .v2_categories import CategoriesRetriever
from .v3_taxonomy import TaxonomyRetriever


class FullEnhancedRetriever(BaseRetriever):
    """V5: Full Enhanced - All metadata features with importance weighting."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
        top_k: int = 10,
        bm25_weight: float = 0.25,
        category_boost: float = 0.15,
        importance_weight: float = 0.15,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        config = RetrieverConfig(
            name="V5: Full Enhanced",
            description="All features + importance weighting",
            top_k=top_k,
            bm25_weight=bm25_weight,
            importance_weight=importance_weight,
        )
        super().__init__(vector_store, embedding_service, config)
        self.category_boost = category_boost
        self.k1 = k1
        self.b = b

        # Create helper retrievers for reusing logic
        self._keywords_retriever = KeywordsRetriever(
            vector_store, embedding_service, top_k, bm25_weight, k1, b
        )
        self._categories_retriever = CategoriesRetriever(
            vector_store, embedding_service, top_k, category_boost
        )
        self._taxonomy_retriever = TaxonomyRetriever(
            vector_store, embedding_service, top_k
        )

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
        """Full enhanced retrieval with all features.

        Pipeline:
        1. Taxonomy expansion (V3)
        2. Dense retrieval on expanded query
        3. BM25 keyword scoring (V1)
        4. Category boosting (V2)
        5. Importance reranking (unique to V5)
        6. Return top_k
        """
        # Step 1: Expand query with taxonomy (V3)
        expansion_terms = self._taxonomy_retriever._expand_query_with_taxonomy(query)
        expanded_query = f"{query} {' '.join(expansion_terms)}" if expansion_terms else query

        # Step 2: Dense retrieval with expanded query
        query_embedding = self._query_to_embedding(expanded_query)
        candidate_k = min(self.config.top_k * 2, 50)

        dense_results_dict = self.vector_store.query(
            query_embedding=query_embedding,
            top_k=candidate_k
        )
        dense_results = self._results_to_retrieval_results(dense_results_dict)

        if not dense_results:
            return []

        # Step 3: BM25 scoring (V1 logic)
        query_terms = self._keywords_retriever._tokenize(query)
        doc_keywords_list = []
        total_length = 0
        doc_freq = Counter()

        for result in dense_results:
            keywords = self._keywords_retriever._get_document_keywords(result.metadata)
            doc_keywords_list.append(keywords)
            total_length += len(keywords)
            unique_terms = set(keywords)
            for term in unique_terms:
                doc_freq[term] += 1

        avg_doc_length = total_length / len(dense_results) if dense_results else 1.0
        total_docs = len(dense_results)

        # Step 4: Predict category (V2 logic)
        predicted_category = self._categories_retriever._predict_category(query)

        # Step 5: Combine all scores
        for i, result in enumerate(dense_results):
            doc_keywords = doc_keywords_list[i]

            # BM25 score
            bm25_score = self._keywords_retriever._compute_bm25_score(
                query_terms,
                doc_keywords,
                avg_doc_length,
                total_docs,
                doc_freq,
            )

            # Normalize BM25
            max_bm25 = len(query_terms) * 5.0
            normalized_bm25 = min(bm25_score / max_bm25, 1.0) if max_bm25 > 0 else 0.0

            # Combine dense + BM25
            dense_score = result.score
            vector_weight = 1.0 - self.config.bm25_weight
            combined_score = (vector_weight * dense_score +
                            self.config.bm25_weight * normalized_bm25)

            # Category boost
            doc_categories = self._categories_retriever._infer_document_category(result.metadata)
            if predicted_category in doc_categories:
                combined_score = min(combined_score + self.category_boost, 1.0)

            result.score = combined_score

        # Step 6: Importance reranking
        reranked = self._rerank_by_importance(dense_results)

        return reranked[:self.config.top_k]
