# rag_runtime/api_server.py
"""
FastAPI server for RAG runtime.

Provides REST API endpoints for:
- Question answering
- Direct search
- Health checks
- Metrics and monitoring
"""
from typing import Optional, List
import logging
import time
import psutil
import torch

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
from .middleware import RequestLoggingMiddleware, RateLimitMiddleware, CacheMiddleware
from .metrics import metrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Track server start time
SERVER_START_TIME = time.time()

# Create FastAPI app
app = FastAPI(
    title="Liquid RAG API",
    description="Production-ready RAG-based question answering using LiquidAI models",
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middleware (order matters - last added runs first)
app.add_middleware(CacheMiddleware, ttl=300, max_size=1000)  # 5 min cache
app.add_middleware(RateLimitMiddleware, calls=100, period=60)  # 100 req/min
app.add_middleware(RequestLoggingMiddleware)


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
    version: str = "0.2.0"
    uptime_seconds: Optional[float] = None
    gpu: Optional[dict] = None
    system: Optional[dict] = None
    api_metrics: Optional[dict] = None


# Endpoints
@app.get("/health")
async def health_check():
    """
    Enhanced health check endpoint.

    Returns detailed system health including:
    - Vector store status
    - GPU availability and memory
    - System resource usage
    - API performance metrics
    - Server uptime
    """
    try:
        uptime = time.time() - SERVER_START_TIME

        # GPU information
        gpu_info = {
            "available": torch.cuda.is_available(),
        }
        if torch.cuda.is_available():
            gpu_info.update({
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name(0),
                "memory_allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
                "memory_reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 2),
                "memory_total_gb": round(
                    torch.cuda.get_device_properties(0).total_memory / 1e9, 2
                ),
            })

        # System resource information
        system_info = {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "cpu_count": psutil.cpu_count(),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_available_gb": round(
                psutil.virtual_memory().available / 1e9, 2
            ),
            "memory_total_gb": round(psutil.virtual_memory().total / 1e9, 2),
        }

        # Vector store stats
        try:
            vector_stats = get_collection_stats()
        except Exception as e:
            vector_stats = {"error": str(e)}

        return {
            "status": "healthy",
            "version": "0.2.0",
            "uptime_seconds": round(uptime, 2),
            "vector_store": vector_stats,
            "gpu": gpu_info,
            "system": system_info,
            "api_metrics": metrics.get_stats(),
        }

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "degraded",
            "error": str(e),
            "version": "0.2.0",
        }


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


@app.get("/metrics")
async def get_metrics():
    """
    Get API performance metrics.

    Returns:
    - Total requests processed
    - Error rate
    - Cache hit rate
    - Response time statistics (avg, min, max, percentiles)
    - Requests per second
    - Server uptime
    """
    return metrics.get_stats()


@app.post("/metrics/reset")
async def reset_metrics():
    """
    Reset all metrics counters.

    Useful for testing or starting fresh monitoring periods.
    """
    metrics.reset()
    return {"status": "metrics reset", "timestamp": time.time()}


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
