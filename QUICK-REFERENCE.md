# LiquidAI Stack - Quick Reference Card

**One-page cheat sheet for common operations**

---

## 🚀 Start Services

```bash
# Enhanced RAG API Server
cd liquid-rag-runtime
uv run python -m rag_runtime.api_server_enhanced
# → http://localhost:8000

# MLflow UI
uv run mlflow ui --port 5000
# → http://localhost:5000

# Original RAG API (no enhancements)
cd liquid-rag-runtime
uv run python -m rag_runtime.api_server
```

---

## 📡 API Endpoints

### Production Endpoints
```bash
GET  /health               # Enhanced health check
GET  /metrics              # Performance metrics
GET  /status               # Quick status
GET  /stats                # Vector store stats
GET  /docs                 # Interactive API docs
```

### RAG Endpoints
```bash
POST /ask                  # Full RAG agent (sophisticated)
POST /ask/simple           # Simple RAG (faster)
POST /search               # Document search only
```

---

## 🧪 Testing

```bash
# Run comprehensive test suite
cd liquid-rag-runtime
uv run python test_enhanced_api.py

# Quick health check
curl http://localhost:8000/health | jq

# Quick metrics check
curl http://localhost:8000/metrics | jq

# Test RAG query
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is LiquidAI?", "fast_mode": true}'
```

---

## 📊 Monitoring

### Check API Metrics
```bash
curl http://localhost:8000/metrics | jq
```

### Important Metrics
- `cache_hit_rate` - Should be >70% after warmup
- `error_rate` - Should be <1%
- `avg_response_time_ms` - Varies by query type

### Check Cache Performance
```bash
# Look for X-Cache header
curl -i -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "test"}' | grep X-Cache

# X-Cache: HIT or MISS
# X-Cache-Age: <seconds since cached>
```

---

## 🧬 Fine-Tuning

### Run Fine-Tuning
```bash
cd liquid-ft-trainer
uv run python -m ft_trainer.train \
  --model models/LFM2-700M \
  --data ../data/ft \
  --epochs 3 \
  --lora
```

### Common Options
```bash
--model <path>          # Model to fine-tune
--epochs <n>            # Number of epochs (default: 3)
--batch-size <n>        # Batch size (default: 4)
--lr <float>            # Learning rate (default: 2e-5)
--lora                  # Use LoRA (default: true)
--no-lora               # Disable LoRA (full fine-tuning)
--4bit                  # Use 4-bit quantization (QLoRA)
```

---

## 📈 MLflow

### Start MLflow UI
```bash
uv run mlflow ui --port 5000
# → http://localhost:5000
```

### View Experiments
```bash
# List all experiments
uv run mlflow experiments list

# List runs in experiment
uv run mlflow runs list --experiment-name liquid-ft-training

# Search for best run
uv run mlflow runs search --experiment-name liquid-ft-training \
  --order-by "metrics.final_train_loss ASC" --max-results 1
```

---

## 🗄️ ETL Pipeline

### Run ETL
```bash
cd liquid-etl-pipeline
uv run python -m etl_pipeline.run_etl

# Or use the CLI entry point
liquid-etl
```

### Check Results
```bash
# Check vector store
ls -la data/vectordb/

# Check fine-tuning data
ls -la data/ft/

# Check processed data
ls -la data/processed/
```

---

## 🔧 Configuration

### Adjust Cache Settings
Edit `liquid-rag-runtime/rag_runtime/api_server_enhanced.py:60`:
```python
CacheMiddleware(ttl=600, max_size=2000)
```

### Adjust Rate Limits
Edit `liquid-rag-runtime/rag_runtime/api_server_enhanced.py:55`:
```python
RateLimitMiddleware(calls=200, period=60)
```

### Disable MLflow
```python
config = FTConfig(use_mlflow=False, ...)
```

---

## 📁 Important Files

### Code
```
liquid-rag-runtime/rag_runtime/api_server_enhanced.py    # Enhanced API
liquid-rag-runtime/rag_runtime/middleware/cache.py       # Caching
liquid-rag-runtime/rag_runtime/middleware/rate_limit.py  # Rate limiting
liquid-ft-trainer/ft_trainer/mlflow_tracking.py          # MLflow
liquid-ft-trainer/ft_trainer/train.py                    # Training
```

### Documentation
```
CLAUDE.md                        # Architecture guide
SETUP.md                         # Setup instructions
IMPLEMENTATION-COMPLETE.md       # Feature summary
CACHING-MLFLOW-GUIDE.md          # Caching & MLflow guide
ENHANCED-API-QUICKSTART.md       # Quick start
```

---

## 🐛 Troubleshooting

### Port Already in Use
```bash
# Find process using port 8000
lsof -i :8000

# Kill it
kill -9 <PID>
```

### Vector Store Empty
```bash
# Run ETL pipeline
cd liquid-etl-pipeline
uv run python -m etl_pipeline.run_etl
```

### MLflow Not Found
```bash
uv add mlflow
```

### Cache Not Working
Check middleware order in `api_server_enhanced.py`:
```python
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(CacheMiddleware)  # Must be last
```

---

## 🎯 Common Tasks

### Add Document to RAG
1. Copy document to `data/raw/`
2. Run ETL: `cd liquid-etl-pipeline && uv run python -m etl_pipeline.run_etl`
3. Restart API server

### Compare Training Runs
1. Start MLflow UI: `uv run mlflow ui --port 5000`
2. Open http://localhost:5000
3. Select multiple runs
4. Click "Compare"

### Check System Health
```bash
curl http://localhost:8000/health | jq .
# Look at: status, gpu, system, api_metrics
```

### Clear Cache
Restart the API server (in-memory cache clears automatically)

---

## 📊 Performance Targets

- **Cache Hit Rate**: >70% after warmup
- **Error Rate**: <1%
- **Response Time (cached)**: <10ms
- **Response Time (uncached)**: 1-3s for RAG
- **Uptime**: >99.9%

---

## 🔑 Key Features

✅ **Request Logging** - Every request tracked
✅ **Rate Limiting** - 100 req/min per IP (configurable)
✅ **Response Caching** - 99.5% faster for cache hits
✅ **Metrics Tracking** - Request/error/timing stats
✅ **Enhanced Health** - GPU, CPU, memory, vector DB
✅ **MLflow Tracking** - Complete experiment history

---

## 📞 Quick Help

- **API Docs**: http://localhost:8000/docs
- **MLflow UI**: http://localhost:5000
- **Architecture**: See `CLAUDE.md`
- **Full Guide**: See `IMPLEMENTATION-COMPLETE.md`

---

**Most common command**:
```bash
cd liquid-rag-runtime && uv run python -m rag_runtime.api_server_enhanced
```

Then open http://localhost:8000/docs 🚀
