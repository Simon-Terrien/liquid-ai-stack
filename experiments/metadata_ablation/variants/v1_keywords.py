"""V1: Keywords retriever - Dense + BM25 keyword matching."""

import sys
import math
from collections import Counter
from typing import List, Dict

sys.path.insert(0, "liquid-shared-core")

from liquid_shared import VectorStore, RetrievalResult, EmbeddingService
from .base import BaseRetriever, RetrieverConfig


class KeywordsRetriever(BaseRetriever):
    """V1: +Keywords - Dense + BM25 keyword matching on tags/entities."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
        top_k: int = 10,
        bm25_weight: float = 0.3,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        config = RetrieverConfig(
            name="V1: +Keywords",
            description="Dense + BM25 keyword matching",
            top_k=top_k,
            bm25_weight=bm25_weight,
        )
        super().__init__(vector_store, embedding_service, config)
        self.k1 = k1  # BM25 term frequency saturation
        self.b = b    # BM25 length normalization

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization: lowercase and split on non-alphanumeric."""
        import re
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def _get_document_keywords(self, metadata: dict) -> List[str]:
        """Extract keywords from metadata tags and entities."""
        keywords = []

        # Extract from tags (comma-separated)
        tags = metadata.get('tags', '')
        if tags:
            keywords.extend([tag.strip().lower() for tag in tags.split(',')])

        # Extract from entities (comma-separated)
        entities = metadata.get('entities', '')
        if entities:
            keywords.extend([entity.strip().lower() for entity in entities.split(',')])

        return keywords

    def _compute_bm25_score(
        self,
        query_terms: List[str],
        doc_keywords: List[str],
        avg_doc_length: float,
        total_docs: int,
        doc_freq: Dict[str, int],
    ) -> float:
        """Compute BM25 score for a document.

        BM25 formula:
        score = Σ IDF(qi) * (f(qi, D) * (k1 + 1)) / (f(qi, D) + k1 * (1 - b + b * |D| / avgdl))

        where:
        - qi: query term
        - D: document
        - f(qi, D): frequency of qi in D
        - |D|: document length
        - avgdl: average document length
        - k1, b: tuning parameters
        """
        if not doc_keywords:
            return 0.0

        doc_length = len(doc_keywords)
        term_freq = Counter(doc_keywords)
        score = 0.0

        for term in query_terms:
            if term not in term_freq:
                continue

            # Compute IDF
            df = doc_freq.get(term, 0)
            if df == 0:
                continue
            idf = math.log((total_docs - df + 0.5) / (df + 0.5) + 1.0)

            # Compute term frequency component
            tf = term_freq[term]
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / avg_doc_length)

            score += idf * (numerator / denominator)

        return score

    def retrieve(self, query: str) -> list[RetrievalResult]:
        """Hybrid search: Dense retrieval + BM25 keyword matching.

        Process:
        1. Dense retrieval to get candidate set (2x top_k)
        2. BM25 scoring on tags/entities
        3. Combine scores: final = (1-w)*dense + w*bm25
        4. Re-rank and return top_k
        """
        # Step 1: Dense retrieval with larger candidate set
        query_embedding = self._query_to_embedding(query)
        candidate_k = min(self.config.top_k * 2, 50)  # Get more candidates for reranking

        dense_results_dict = self.vector_store.query(
            query_embedding=query_embedding,
            top_k=candidate_k
        )
        dense_results = self._results_to_retrieval_results(dense_results_dict)

        if not dense_results:
            return []

        # Step 2: Prepare BM25 scoring
        query_terms = self._tokenize(query)

        # Collect document keywords and compute statistics
        doc_keywords_list = []
        total_length = 0
        doc_freq = Counter()  # Document frequency for each term

        for result in dense_results:
            keywords = self._get_document_keywords(result.metadata)
            doc_keywords_list.append(keywords)
            total_length += len(keywords)

            # Update document frequency
            unique_terms = set(keywords)
            for term in unique_terms:
                doc_freq[term] += 1

        avg_doc_length = total_length / len(dense_results) if dense_results else 1.0
        total_docs = len(dense_results)

        # Step 3: Compute BM25 scores and combine with dense scores
        for i, result in enumerate(dense_results):
            doc_keywords = doc_keywords_list[i]

            # Compute BM25 score
            bm25_score = self._compute_bm25_score(
                query_terms,
                doc_keywords,
                avg_doc_length,
                total_docs,
                doc_freq,
            )

            # Normalize BM25 score to [0, 1] range (approximate)
            # Use query length as max possible score approximation
            max_bm25 = len(query_terms) * 5.0  # Rough upper bound
            normalized_bm25 = min(bm25_score / max_bm25, 1.0) if max_bm25 > 0 else 0.0

            # Combine scores
            dense_score = result.score
            vector_weight = 1.0 - self.config.bm25_weight
            combined_score = (vector_weight * dense_score +
                            self.config.bm25_weight * normalized_bm25)

            result.score = combined_score

        # Step 4: Re-rank by combined score and return top_k
        reranked = sorted(dense_results, key=lambda r: r.score, reverse=True)
        return reranked[:self.config.top_k]
