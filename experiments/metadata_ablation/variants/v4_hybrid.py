"""V4: Hybrid retriever - Keywords + Categories combined."""

import sys
import math
from collections import Counter
from typing import List, Dict, Set

sys.path.insert(0, "liquid-shared-core")

from liquid_shared import VectorStore, RetrievalResult, EmbeddingService
from .base import BaseRetriever, RetrieverConfig
from .v1_keywords import KeywordsRetriever
from .v2_categories import CategoriesRetriever


class HybridRetriever(BaseRetriever):
    """V4: Keywords+Categories - Combines BM25 keywords with category boosting."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
        top_k: int = 10,
        bm25_weight: float = 0.3,
        category_boost: float = 0.15,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        config = RetrieverConfig(
            name="V4: Keywords+Categories",
            description="Hybrid (dense + BM25) + category filtering",
            top_k=top_k,
            bm25_weight=bm25_weight,
        )
        super().__init__(vector_store, embedding_service, config)
        self.category_boost = category_boost
        self.k1 = k1  # BM25 parameters
        self.b = b

        # Create helper retrievers for reusing logic
        self._keywords_retriever = KeywordsRetriever(
            vector_store, embedding_service, top_k, bm25_weight, k1, b
        )
        self._categories_retriever = CategoriesRetriever(
            vector_store, embedding_service, top_k, category_boost
        )

    def retrieve(self, query: str) -> list[RetrievalResult]:
        """Hybrid search combining BM25 keywords and category boosting.

        Process:
        1. Dense retrieval (candidate set)
        2. BM25 scoring on keywords (from V1)
        3. Predict category (from V2)
        4. Boost category matches (from V2)
        5. Re-rank and return top_k
        """
        # Step 1: Dense retrieval with larger candidate set
        query_embedding = self._query_to_embedding(query)
        candidate_k = min(self.config.top_k * 2, 50)

        dense_results_dict = self.vector_store.query(
            query_embedding=query_embedding,
            top_k=candidate_k
        )
        dense_results = self._results_to_retrieval_results(dense_results_dict)

        if not dense_results:
            return []

        # Step 2: BM25 scoring (reuse V1 logic)
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

        # Step 3: Predict category (reuse V2 logic)
        predicted_category = self._categories_retriever._predict_category(query)

        # Step 4: Combine BM25 + dense + category boost
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

        # Step 5: Re-rank by combined score
        reranked = sorted(dense_results, key=lambda r: r.score, reverse=True)
        return reranked[:self.config.top_k]
