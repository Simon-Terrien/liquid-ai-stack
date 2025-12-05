# rag_runtime/api_server.py
"""
FastAPI server for RAG runtime.

Provides REST API endpoints for:
- Question answering
- Direct search
- Health checks
"""
from typing import Optional, List
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .rag_agent import (
    get_rag_agent,
    ask_sync,
    simple_rag,
    RAGDependencies,
)
from .tools.retrieval import (
    retrieve_chunks,
    hybrid_search,
    get_collection_stats,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Liquid RAG API",
    description="RAG-based question answering using LiquidAI models",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class AskRequest(BaseModel):
    """Request for /ask endpoint."""
    question: str = Field(description="The question to answer")
    max_context_chunks: int = Field(default=5, ge=1, le=20)
    use_hybrid_search: bool = Field(default=True)
    fast_mode: bool = Field(default=True, description="Use faster model (700M vs 1.2B)")


class AskResponse(BaseModel):
    """Response from /ask endpoint."""
    answer: str
    sources: List[str] = []
    confidence: Optional[float] = None
    context_used: int = 0


class SearchRequest(BaseModel):
    """Request for /search endpoint."""
    query: str = Field(description="Search query")
    top_k: int = Field(default=5, ge=1, le=20)
    hybrid: bool = Field(default=True, description="Use hybrid search")


class SearchResultItem(BaseModel):
    """A single search result."""
    chunk_id: str
    text: str
    score: float
    source: str
    section: Optional[str] = None


class SearchResponse(BaseModel):
    """Response from /search endpoint."""
    results: List[SearchResultItem]
    total: int
    query: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    vector_store: dict
    version: str = "0.1.0"


# Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        stats = get_collection_stats()
        return HealthResponse(
            status="healthy",
            vector_store=stats,
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="degraded",
            vector_store={"error": str(e)},
        )


@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    Ask a question and get a RAG-based answer.
    
    The system will:
    1. Search for relevant document chunks
    2. Use the context to generate an answer
    3. Return the answer with source citations
    """
    try:
        logger.info(f"Question: {request.question[:100]}...")
        
        # Get appropriate agent
        agent = get_rag_agent(fast_mode=request.fast_mode)
        
        # Configure dependencies
        deps = RAGDependencies(
            max_context_chunks=request.max_context_chunks,
            use_hybrid_search=request.use_hybrid_search,
        )
        
        # Get answer
        response = ask_sync(request.question, agent=agent, deps=deps)
        
        return AskResponse(
            answer=response.answer,
            sources=response.sources,
            confidence=response.confidence,
            context_used=response.context_used,
        )
        
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask/simple", response_model=AskResponse)
async def ask_simple(request: AskRequest):
    """
    Simple RAG without full agent infrastructure.
    
    Faster but less sophisticated than /ask.
    """
    try:
        answer = simple_rag(
            request.question,
            top_k=request.max_context_chunks,
        )
        
        return AskResponse(
            answer=answer,
            sources=[],
            context_used=request.max_context_chunks,
        )
        
    except Exception as e:
        logger.error(f"Error in simple RAG: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """
    Search for relevant document chunks.
    
    Returns chunks sorted by relevance score.
    """
    try:
        if request.hybrid:
            results = hybrid_search(request.query, top_k=request.top_k)
        else:
            results = retrieve_chunks(request.query, top_k=request.top_k)
        
        items = []
        for r in results:
            items.append(SearchResultItem(
                chunk_id=r.chunk_id,
                text=r.text,
                score=r.score,
                source=r.metadata.get("source", "unknown"),
                section=r.metadata.get("section_title"),
            ))
        
        return SearchResponse(
            results=items,
            total=len(items),
            query=request.query,
        )
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get vector store statistics."""
    return get_collection_stats()


# CLI runner
def main():
    """Run the API server."""
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser(description="Run RAG API server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    uvicorn.run(
        "rag_runtime.api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
