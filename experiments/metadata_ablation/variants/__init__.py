"""Retrieval variant implementations for metadata ablation study.

This package contains 6 retrieval variants (V0-V5) that progressively add
enhanced metadata features to evaluate their impact on RAG retrieval quality.

Variants:
- V0 (BaselineRetriever): Dense retrieval only
- V1 (KeywordsRetriever): Dense + BM25 keyword matching
- V2 (CategoriesRetriever): Dense + category filtering
- V3 (TaxonomyRetriever): Dense + taxonomy-based query expansion
- V4 (HybridRetriever): Keywords + Categories combined
- V5 (FullEnhancedRetriever): All features + importance weighting

Usage:
    from metadata_ablation.variants import create_retriever

    retriever = create_retriever("v0", vector_store, embedding_service, top_k=10)
    results = retriever.retrieve("What are adversarial attacks?")
"""

from .base import BaseRetriever, RetrieverConfig
from .v0_baseline import BaselineRetriever
from .v1_keywords import KeywordsRetriever
from .v2_categories import CategoriesRetriever
from .v3_taxonomy import TaxonomyRetriever
from .v4_hybrid import HybridRetriever
from .v5_full_enhanced import FullEnhancedRetriever
from .factory import create_retriever

__all__ = [
    # Base classes
    "BaseRetriever",
    "RetrieverConfig",
    # Variant implementations
    "BaselineRetriever",
    "KeywordsRetriever",
    "CategoriesRetriever",
    "TaxonomyRetriever",
    "HybridRetriever",
    "FullEnhancedRetriever",
    # Factory function
    "create_retriever",
]
