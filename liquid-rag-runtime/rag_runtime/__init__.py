# rag_runtime/__init__.py
"""
Liquid RAG Runtime - Fast inference for retrieval-augmented generation.

Features:
- LiquidAI-powered answer generation (700M/1.2B models)
- Semantic and hybrid search
- Tool-based context retrieval
- FastAPI server for REST API
- CLI for interactive use

Usage:
    from rag_runtime import ask, ask_sync, simple_rag
    
    # Async
    response = await ask("What is GDPR?")
    print(response.answer)
    
    # Sync
    response = ask_sync("What is GDPR?")
    
    # Simple (no agent overhead)
    answer = simple_rag("What is GDPR?")
"""

from .rag_agent import (
    create_rag_agent,
    get_rag_agent,
    ask,
    ask_sync,
    simple_rag,
    RAGDependencies,
    RAGResponse,
)

from .tools import (
    retrieve_chunks,
    hybrid_search,
    search_by_metadata,
    get_chunk_by_id,
    get_collection_stats,
    search_tool,
    search_with_context_tool,
)

__version__ = "0.1.0"

__all__ = [
    # Agent
    "create_rag_agent",
    "get_rag_agent",
    "ask",
    "ask_sync",
    "simple_rag",
    "RAGDependencies",
    "RAGResponse",
    # Tools
    "retrieve_chunks",
    "hybrid_search",
    "search_by_metadata",
    "get_chunk_by_id",
    "get_collection_stats",
    "search_tool",
    "search_with_context_tool",
]
