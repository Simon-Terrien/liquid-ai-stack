"""Factory function for creating retriever instances."""

import sys

sys.path.insert(0, "liquid-shared-core")

from liquid_shared import VectorStore, EmbeddingService
from .base import BaseRetriever
from .v0_baseline import BaselineRetriever
from .v1_keywords import KeywordsRetriever
from .v2_categories import CategoriesRetriever
from .v3_taxonomy import TaxonomyRetriever
from .v4_hybrid import HybridRetriever
from .v5_full_enhanced import FullEnhancedRetriever


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
