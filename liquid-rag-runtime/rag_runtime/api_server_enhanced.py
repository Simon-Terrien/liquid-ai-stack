"""
Enhanced FastAPI server for RAG runtime with production features.

Enhancements:
- Request logging middleware
- Rate limiting (100 requests/minute)
- Performance metrics
- Enhanced health check
- API documentation
"""
import logging
import time
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import middleware
from rag_runtime.middleware import RequestLoggingMiddleware, RateLimitMiddleware, CacheMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastAPI app with enhanced metadata
app = FastAPI(
    title="Liquid RAG API (Enhanced)",
    description="""
    Production-ready RAG API using LiquidAI models with:
    - Request logging
    - Rate limiting  
    - Performance metrics
    - Health monitoring
    """,
    version="1.0.0",
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

# Add custom middleware (order matters - cache first for early returns)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    RateLimitMiddleware,
    calls=100,  # 100 requests
    period=60,  # per 60 seconds
)
app.add_middleware(
    CacheMiddleware,
    ttl=300,  # 5 minutes
    max_size=1000,  # Max 1000 cached items
)

# Track server start time
start_time = time.time()

# Simple metrics tracking
class Metrics:
    def __init__(self):
        self.total_requests = 0
        self.total_errors = 0
        self.total_cache_hits = 0
        self.total_cache_misses = 0
        self.response_times = []

    def record_request(self, duration: float, error: bool = False, cache_hit: bool = False):
        self.total_requests += 1
        if error:
            self.total_errors += 1
        if cache_hit:
            self.total_cache_hits += 1
        else:
            self.total_cache_misses += 1
        self.response_times.append(duration)
        # Keep only last 1000 response times
        if len(self.response_times) > 1000:
            self.response_times = self.response_times[-1000:]

    def get_stats(self) -> dict:
        total_cacheable = self.total_cache_hits + self.total_cache_misses
        return {
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "error_rate": (self.total_errors / self.total_requests * 100) if self.total_requests > 0 else 0,
            "cache_hits": self.total_cache_hits,
            "cache_misses": self.total_cache_misses,
            "cache_hit_rate": (self.total_cache_hits / total_cacheable * 100) if total_cacheable > 0 else 0,
            "avg_response_time_ms": (sum(self.response_times) / len(self.response_times) * 1000) if self.response_times else 0,
            "min_response_time_ms": (min(self.response_times) * 1000) if self.response_times else 0,
            "max_response_time_ms": (max(self.response_times) * 1000) if self.response_times else 0,
        }

metrics = Metrics()


# Request/Response models (same as original)
class AskRequest(BaseModel):
    question: str = Field(description="The question to answer")
    max_context_chunks: int = Field(default=5, ge=1, le=20)
    use_hybrid_search: bool = Field(default=True)
    fast_mode: bool = Field(default=True, description="Use faster model (700M vs 1.2B)")


class AskResponse(BaseModel):
    answer: str
    sources: List[str] = []
    confidence: Optional[float] = None
    context_used: int = 0


class SearchRequest(BaseModel):
    query: str = Field(description="Search query")
    top_k: int = Field(default=5, ge=1, le=20)
    hybrid: bool = Field(default=True, description="Use hybrid search")


class SearchResultItem(BaseModel):
    chunk_id: str
    text: str
    score: float
    source: str
    section: Optional[str] = None


class SearchResponse(BaseModel):
    results: List[SearchResultItem]
    total: int
    query: str


# Enhanced health endpoint
@app.get("/health")
async def health_check():
    """
    Enhanced health check with system metrics.

    Returns:
        - status: Overall health status
        - uptime: Server uptime in seconds
        - models: Model availability status
        - metrics: Request metrics
        - vector_store: Vector store statistics
    """
    try:
        import torch
        import psutil

        uptime = time.time() - start_time

        health_data = {
            "status": "healthy",
            "uptime_seconds": round(uptime, 2),
            "uptime_human": f"{int(uptime // 3600)}h {int((uptime % 3600) // 60)}m",
            "gpu": {
                "available": torch.cuda.is_available(),
            },
            "system": {
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "memory_percent": psutil.virtual_memory().percent,
            },
            "api_metrics": metrics.get_stats(),
            "vector_store": get_collection_stats(),
        }

        if torch.cuda.is_available():
            health_data["gpu"]["memory_used_gb"] = round(torch.cuda.memory_allocated() / 1e9, 2)
            health_data["gpu"]["memory_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
            health_data["gpu"]["memory_percent"] = round(
                torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory * 100, 2
            )

        return health_data

    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "degraded",
            "error": str(e),
        }


# Metrics endpoint
@app.get("/metrics")
async def get_metrics():
    """
    Get detailed API metrics.
    
    Returns performance statistics for monitoring.
    """
    return {
        **metrics.get_stats(),
        "uptime_seconds": round(time.time() - start_time, 2),
    }


# Status endpoint
@app.get("/status")
async def get_status():
    """Quick status check (faster than /health)"""
    return {
        "status": "ok",
        "uptime": round(time.time() - start_time, 2),
    }


# Import RAG agent and retrieval tools
from rag_runtime.rag_agent import get_rag_agent, ask_sync, simple_rag, RAGDependencies
from rag_runtime.tools.retrieval import retrieve_chunks, hybrid_search, get_collection_stats


# Enhanced ask endpoint with RAG agent integration
@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    Ask a question and get a RAG-based answer.

    The system will:
    1. Search for relevant document chunks
    2. Use the context to generate an answer with LiquidAI model
    3. Return the answer with source citations

    Uses full Pydantic AI agent with retrieval tools.
    """
    start = time.time()

    try:
        logger.info(f"Question: {request.question[:100]}...")

        # Get appropriate agent based on fast_mode
        agent = get_rag_agent(fast_mode=request.fast_mode)

        # Configure dependencies
        deps = RAGDependencies(
            max_context_chunks=request.max_context_chunks,
            use_hybrid_search=request.use_hybrid_search,
        )

        # Get answer using RAG agent
        response = ask_sync(request.question, agent=agent, deps=deps)

        duration = time.time() - start
        metrics.record_request(duration, error=False)

        return AskResponse(
            answer=response.answer,
            sources=response.sources,
            confidence=response.confidence,
            context_used=response.context_used,
        )

    except Exception as e:
        duration = time.time() - start
        metrics.record_request(duration, error=True)
        logger.error(f"Error answering question: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask/simple", response_model=AskResponse)
async def ask_simple(request: AskRequest):
    """
    Simple RAG without full agent infrastructure.

    Faster but less sophisticated than /ask.
    Direct retrieval + generation without agent overhead.
    """
    start = time.time()

    try:
        answer = simple_rag(
            request.question,
            top_k=request.max_context_chunks,
        )

        duration = time.time() - start
        metrics.record_request(duration, error=False)

        return AskResponse(
            answer=answer,
            sources=[],
            context_used=request.max_context_chunks,
        )

    except Exception as e:
        duration = time.time() - start
        metrics.record_request(duration, error=True)
        logger.error(f"Error in simple RAG: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """
    Search for relevant document chunks.

    Returns chunks sorted by relevance score without generation.
    """
    start = time.time()

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

        duration = time.time() - start
        metrics.record_request(duration, error=False)

        return SearchResponse(
            results=items,
            total=len(items),
            query=request.query,
        )

    except Exception as e:
        duration = time.time() - start
        metrics.record_request(duration, error=True)
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_vector_stats():
    """Get vector store statistics."""
    return get_collection_stats()


# CLI runner
def main():
    """Run the enhanced API server."""
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Enhanced RAG API server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    logger.info(f"Starting Enhanced RAG API server on {args.host}:{args.port}")
    logger.info(f"📚 API Docs: http://{args.host}:{args.port}/docs")
    logger.info(f"🏥 Health: http://{args.host}:{args.port}/health")
    logger.info(f"📊 Metrics: http://{args.host}:{args.port}/metrics")
    
    uvicorn.run(
        "rag_runtime.api_server_enhanced:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
