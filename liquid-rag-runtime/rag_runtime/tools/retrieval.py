# rag_runtime/tools/retrieval.py
"""
Retrieval tools for RAG pipeline.

Provides MCP-compatible tools for:
- Semantic search
- Hybrid search (semantic + keyword)
- Reranking
"""
from typing import List, Optional, Dict, Any
import logging

from pydantic import BaseModel, Field

from liquid_shared import (
    VectorStore,
    EmbeddingService,
    RetrievalResult,
    DATA_DIR,
)

logger = logging.getLogger(__name__)

# Default paths
VDB_DIR = DATA_DIR / "vectordb"

# Global services (lazy loaded)
_embedder: Optional[EmbeddingService] = None
_vstore: Optional[VectorStore] = None


def get_embedder() -> EmbeddingService:
    """Get or create the embedding service."""
    global _embedder
    if _embedder is None:
        _embedder = EmbeddingService()
    return _embedder


def get_vstore() -> VectorStore:
    """Get or create the vector store."""
    global _vstore
    if _vstore is None:
        _vstore = VectorStore(VDB_DIR)
    return _vstore


class SearchQuery(BaseModel):
    """Input for search operations."""
    query: str = Field(description="Search query text")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results")
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadata filters (e.g., {'source': 'doc.pdf'})"
    )


class SearchResult(BaseModel):
    """A single search result."""
    chunk_id: str
    text: str
    score: float
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    """Response from search operations."""
    results: List[SearchResult]
    query: str
    total_found: int


def retrieve_chunks(
    query: str,
    top_k: int = 5,
    filters: Optional[Dict[str, Any]] = None,
    embedder: Optional[EmbeddingService] = None,
    vstore: Optional[VectorStore] = None,
) -> List[RetrievalResult]:
    """
    Retrieve relevant chunks using semantic search.
    
    Args:
        query: Search query text
        top_k: Number of results to return
        filters: Optional metadata filters
        embedder: Optional custom embedder
        vstore: Optional custom vector store
        
    Returns:
        List of RetrievalResult objects sorted by relevance
    """
    embedder = embedder or get_embedder()
    vstore = vstore or get_vstore()
    
    # Generate query embedding
    query_embedding = embedder.encode_single(query)
    
    # Search vector store
    results = vstore.query(
        query_embedding=query_embedding,
        top_k=top_k,
        where=filters,
    )
    
    # Convert to RetrievalResult objects
    retrieval_results = []
    
    if results.get("ids") and results["ids"][0]:
        ids = results["ids"][0]
        documents = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        
        for i, chunk_id in enumerate(ids):
            # Convert distance to similarity score (assuming cosine distance)
            score = 1 - distances[i] if i < len(distances) else 0.0
            
            retrieval_results.append(RetrievalResult(
                chunk_id=chunk_id,
                text=documents[i] if i < len(documents) else "",
                score=max(0, min(1, score)),  # Clamp to [0, 1]
                metadata=metadatas[i] if i < len(metadatas) else {},
            ))
    
    logger.info(f"Retrieved {len(retrieval_results)} chunks for query: {query[:50]}...")
    
    return retrieval_results


def hybrid_search(
    query: str,
    top_k: int = 5,
    semantic_weight: float = 0.7,
    keyword_weight: float = 0.3,
    filters: Optional[Dict[str, Any]] = None,
) -> List[RetrievalResult]:
    """
    Hybrid search combining semantic and keyword matching.
    
    Args:
        query: Search query text
        top_k: Number of results to return
        semantic_weight: Weight for semantic similarity (0-1)
        keyword_weight: Weight for keyword matching (0-1)
        filters: Optional metadata filters
        
    Returns:
        List of RetrievalResult objects with combined scores
    """
    embedder = get_embedder()
    vstore = get_vstore()
    
    # Semantic search
    semantic_results = retrieve_chunks(
        query=query,
        top_k=top_k * 2,  # Fetch more for reranking
        filters=filters,
        embedder=embedder,
        vstore=vstore,
    )
    
    # Simple keyword matching (using ChromaDB's where_document)
    query_terms = query.lower().split()
    
    # Score adjustment based on keyword presence
    for result in semantic_results:
        text_lower = result.text.lower()
        keyword_score = sum(1 for term in query_terms if term in text_lower) / len(query_terms)
        
        # Combine scores
        result.score = (
            semantic_weight * result.score +
            keyword_weight * keyword_score
        )
    
    # Re-sort by combined score
    semantic_results.sort(key=lambda x: x.score, reverse=True)
    
    return semantic_results[:top_k]


def search_by_metadata(
    filters: Dict[str, Any],
    limit: int = 10,
) -> List[RetrievalResult]:
    """
    Search chunks by metadata filters only.
    
    Args:
        filters: Metadata filters (e.g., {'source': 'doc.pdf', 'importance': 8})
        limit: Maximum results
        
    Returns:
        List of matching chunks
    """
    vstore = get_vstore()
    
    results = vstore.get(where=filters, limit=limit)
    
    retrieval_results = []
    if results.get("ids"):
        for i, chunk_id in enumerate(results["ids"]):
            retrieval_results.append(RetrievalResult(
                chunk_id=chunk_id,
                text=results.get("documents", [])[i] if i < len(results.get("documents", [])) else "",
                score=1.0,  # No ranking for metadata search
                metadata=results.get("metadatas", [])[i] if i < len(results.get("metadatas", [])) else {},
            ))
    
    return retrieval_results


def get_chunk_by_id(chunk_id: str) -> Optional[RetrievalResult]:
    """Get a specific chunk by ID."""
    vstore = get_vstore()
    
    results = vstore.get(ids=[chunk_id])
    
    if results.get("ids") and results["ids"]:
        return RetrievalResult(
            chunk_id=results["ids"][0],
            text=results.get("documents", [""])[0],
            score=1.0,
            metadata=results.get("metadatas", [{}])[0],
        )
    
    return None


def get_collection_stats() -> Dict[str, Any]:
    """Get statistics about the vector store."""
    vstore = get_vstore()
    
    return {
        "total_documents": vstore.count(),
        "collection_name": vstore.collection_name,
        "persist_directory": str(vstore.persist_dir) if vstore.persist_dir else "in-memory",
    }


# MCP-style tool wrappers for Pydantic AI

def search_tool(query: str, top_k: int = 5) -> List[str]:
    """
    MCP tool: Search for relevant document chunks.
    
    Args:
        query: What to search for
        top_k: Number of results (default 5)
        
    Returns:
        List of relevant text chunks
    """
    results = retrieve_chunks(query, top_k)
    return [r.text for r in results]


def search_with_context_tool(query: str, top_k: int = 5) -> str:
    """
    MCP tool: Search and format results with context.
    
    Returns formatted string suitable for LLM context.
    """
    results = retrieve_chunks(query, top_k)
    
    if not results:
        return "No relevant documents found."
    
    formatted = []
    for i, r in enumerate(results, 1):
        meta = r.metadata
        source = meta.get("source", "unknown")
        section = meta.get("section_title", "")
        
        header = f"[Source {i}: {source}"
        if section:
            header += f" - {section}"
        header += f"] (relevance: {r.score:.2f})"
        
        formatted.append(f"{header}\n{r.text}")
    
    return "\n\n---\n\n".join(formatted)
