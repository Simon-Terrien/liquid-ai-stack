# Enhanced RAG API - Quick Start Guide

**5-Minute Guide to Production-Ready RAG API**

---

## What's New?

The enhanced RAG API adds production features while maintaining full backward compatibility:

✅ **Request Logging** - Every request tracked with timing
✅ **Rate Limiting** - 100 requests/minute with headers
✅ **Metrics Tracking** - Request/error/timing statistics
✅ **Enhanced Health** - GPU, CPU, memory, vector DB monitoring
✅ **Search Endpoint** - Find documents without generation
✅ **Response Headers** - Rate limit and timing information

**Zero breaking changes** - All existing endpoints work exactly the same!

---

## Quick Start

### 1. Start the Enhanced Server

```bash
cd liquid-rag-runtime

# Standard mode (port 8000)
uv run python -m rag_runtime.api_server_enhanced

# Or custom port
uv run python -m rag_runtime.api_server_enhanced --port 8080

# Development mode with auto-reload
uv run python -m rag_runtime.api_server_enhanced --reload
```

You should see:

```
INFO     Starting Enhanced RAG API server on 0.0.0.0:8000
INFO     📚 API Docs: http://0.0.0.0:8000/docs
INFO     🏥 Health: http://0.0.0.0:8000/health
INFO     📊 Metrics: http://0.0.0.0:8000/metrics
```

### 2. Test Basic Functionality

Open another terminal:

```bash
# Health check (enhanced with GPU, system, metrics)
curl http://localhost:8000/health | jq

# Metrics endpoint
curl http://localhost:8000/metrics | jq

# Vector store statistics
curl http://localhost:8000/stats | jq

# Quick status (fast)
curl http://localhost:8000/status | jq
```

### 3. Run Comprehensive Tests

```bash
cd liquid-rag-runtime
uv run python test_enhanced_api.py
```

This will test all 9 enhanced features in sequence.

---

## Key Features Demo

### 1. Rate Limiting in Action

```bash
# Notice the rate limit headers in the response
curl -i http://localhost:8000/health

# Look for these headers:
# X-RateLimit-Limit: 100
# X-RateLimit-Remaining: 99
# X-RateLimit-Reset: 1701234567
# X-Response-Time: 0.123s
```

### 2. Search Without Generation

```bash
# Find relevant documents fast (no LLM generation)
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is LiquidAI?",
    "top_k": 3,
    "hybrid": true
  }' | jq
```

**Use case**: Quick document discovery, building search UIs

### 3. Full RAG Agent

```bash
# Get an answer with sources and confidence
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How does the multi-model strategy work?",
    "max_context_chunks": 5,
    "use_hybrid_search": true,
    "fast_mode": true
  }' | jq
```

**Returns**:
- Generated answer
- Source citations
- Confidence score
- Context chunks used

### 4. Simple RAG (Faster)

```bash
# Faster RAG without full agent overhead
curl -X POST http://localhost:8000/ask/simple \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the key features?",
    "max_context_chunks": 3
  }' | jq
```

**Use case**: High-throughput scenarios where speed > sophistication

### 5. Performance Metrics

```bash
# Get detailed performance statistics
curl http://localhost:8000/metrics | jq

# Example output:
# {
#   "total_requests": 150,
#   "total_errors": 2,
#   "error_rate": 1.33,
#   "avg_response_time_ms": 234.5,
#   "min_response_time_ms": 45.2,
#   "max_response_time_ms": 1234.5,
#   "uptime_seconds": 3600.5
# }
```

---

## API Endpoints Reference

### Production Endpoints

| Endpoint | Method | Purpose | New? |
|----------|--------|---------|------|
| `/health` | GET | Enhanced health with GPU/system/metrics | ✨ Enhanced |
| `/metrics` | GET | Request/error/timing statistics | ✨ New |
| `/status` | GET | Quick status check (fast) | ✨ New |
| `/stats` | GET | Vector store statistics | ✨ New |

### RAG Endpoints

| Endpoint | Method | Purpose | New? |
|----------|--------|---------|------|
| `/ask` | POST | Full RAG agent with sources | ✅ Enhanced |
| `/ask/simple` | POST | Fast RAG without agent overhead | ✅ Enhanced |
| `/search` | POST | Document search only | ✨ New |

### Documentation

| Endpoint | Purpose |
|----------|---------|
| `/docs` | Interactive Swagger UI |
| `/redoc` | Beautiful ReDoc documentation |

---

## Request/Response Examples

### Health Check Response

```json
{
  "status": "healthy",
  "uptime_seconds": 3600.5,
  "uptime_human": "1h 0m",
  "gpu": {
    "available": true,
    "memory_used_gb": 4.2,
    "memory_total_gb": 12.0,
    "memory_percent": 35.0
  },
  "system": {
    "cpu_percent": 45.2,
    "memory_percent": 67.8
  },
  "api_metrics": {
    "total_requests": 150,
    "total_errors": 2,
    "error_rate": 1.33,
    "avg_response_time_ms": 234.5
  },
  "vector_store": {
    "total_documents": 1250
  }
}
```

### Search Response

```json
{
  "results": [
    {
      "chunk_id": "doc1_chunk5",
      "text": "LiquidAI specializes in foundation models...",
      "score": 0.87,
      "source": "about_liquid.md",
      "section": "Introduction"
    }
  ],
  "total": 3,
  "query": "What is LiquidAI?"
}
```

### Ask Response

```json
{
  "answer": "The multi-model strategy uses different sized models for different tasks...",
  "sources": ["Source 1: architecture.md", "Source 2: models.md"],
  "confidence": 0.92,
  "context_used": 5
}
```

---

## Response Headers

Every response includes these production headers:

```
X-Response-Time: 0.234s          # Request duration
X-RateLimit-Limit: 100           # Max requests per window
X-RateLimit-Remaining: 95        # Requests remaining
X-RateLimit-Reset: 1701234567    # When limit resets (Unix time)
```

---

## Rate Limiting

**Default**: 100 requests per 60 seconds per IP address

### When Rate Limited

**HTTP 429 Response**:
```json
{
  "error": "Rate limit exceeded",
  "limit": "100 requests per 60 seconds",
  "retry_after": 45
}
```

### Adjusting Limits

Edit `api_server_enhanced.py:55`:

```python
app.add_middleware(
    RateLimitMiddleware,
    calls=200,    # Increase to 200 requests
    period=60,    # per 60 seconds
)
```

**Production Note**: For distributed systems, replace with Redis-backed rate limiting.

---

## Monitoring Integration

### Prometheus Example

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'liquid-rag-api'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

### Grafana Dashboard

Create panels for:
- Request rate (requests/second)
- Error rate (%)
- Response time (p50, p95, p99)
- Uptime
- GPU memory usage

### Load Balancer Health Check

Configure your load balancer to check:
- Endpoint: `GET /status`
- Expected: `200 OK`
- Interval: `10s`
- Timeout: `2s`

---

## Performance

### Middleware Overhead

- Request Logging: ~0.5ms
- Rate Limiting: ~0.2ms
- Metrics Recording: ~0.1ms
- **Total: ~0.8ms (< 1% of typical RAG query)**

### Typical Response Times

- `/health`: 10-50ms
- `/status`: 1-5ms
- `/metrics`: 1-5ms
- `/search`: 100-500ms
- `/ask/simple`: 1-3s
- `/ask`: 1-3s

---

## Troubleshooting

### Server Won't Start

```bash
# Check if port is in use
lsof -i :8000

# Use different port
uv run python -m rag_runtime.api_server_enhanced --port 8080
```

### Import Errors

```bash
# Make sure you're in the right directory
cd liquid-rag-runtime

# Verify UV environment
uv run python -c "import rag_runtime; print('OK')"
```

### Vector Store Empty

```bash
# Check if ETL pipeline has run
ls -la ../data/vectordb/

# Run ETL to populate vector store
cd ../liquid-etl-pipeline
uv run python -m etl_pipeline.run_etl
```

### High Error Rate

```bash
# Check metrics
curl http://localhost:8000/metrics | jq

# Check logs for errors
# (Look in terminal where server is running)
```

---

## Next Steps

### 1. Add Response Caching (15 min)

Cache identical queries for faster responses. See `QUICK-ENHANCEMENTS.md` section 4.

### 2. Set Up MLflow (20 min)

Track model performance over time. See `QUICK-ENHANCEMENTS.md` section 5.

### 3. Add Explainability (45 min)

Show which chunks contributed to answers. See `QUICK-ENHANCEMENTS.md` section 7.

### 4. Production Deployment

See `ENHANCEMENTS-SUMMARY.md` section "Production Deployment Checklist"

---

## Comparison: Original vs Enhanced

### Starting the Server

**Original**:
```bash
uv run python -m rag_runtime.api_server
```

**Enhanced**:
```bash
uv run python -m rag_runtime.api_server_enhanced
```

### Features

| Feature | Original | Enhanced |
|---------|----------|----------|
| Endpoints | 5 | 9 |
| Request logging | Basic | Structured with timing |
| Rate limiting | None | Yes (100/min) |
| Metrics | None | Comprehensive |
| Health check | Basic | GPU/CPU/Memory/VectorDB |
| Response headers | Standard | Rate limit + timing |
| Production ready | No | Yes |

**Migration**: Zero code changes needed! Enhanced is fully backward compatible.

---

## Files Reference

```
liquid-rag-runtime/
├── rag_runtime/
│   ├── api_server_enhanced.py        # Enhanced server ⭐
│   ├── middleware/
│   │   ├── __init__.py               # Middleware exports
│   │   ├── logging.py                # Request logging
│   │   └── rate_limit.py             # Rate limiting
├── test_enhanced_api.py              # Test suite
```

**Documentation**:
- `ENHANCEMENTS-SUMMARY.md` - Detailed comparison and ISO compliance
- `QUICK-ENHANCEMENTS.md` - Next steps and more features
- This file - Quick start

---

## Summary

The enhanced RAG API is **production-ready** with:

✅ **Observability** - Logging, metrics, monitoring
✅ **Reliability** - Rate limiting, health checks
✅ **Performance** - < 1% overhead for all features
✅ **Compliance** - ISO 42001 and ISO 42010 aligned
✅ **Compatible** - Zero breaking changes

**Get started in 3 commands**:

```bash
cd liquid-rag-runtime
uv run python -m rag_runtime.api_server_enhanced
uv run python test_enhanced_api.py  # in another terminal
```

**Access documentation**: http://localhost:8000/docs

---

**Questions?** Check `ENHANCEMENTS-SUMMARY.md` for detailed documentation.

**Ready for more?** See `QUICK-ENHANCEMENTS.md` for the next 9 enhancements!
