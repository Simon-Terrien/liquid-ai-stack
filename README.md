# LiquidAI Multi-Agent Stack

A production-ready, multi-repo architecture for building intelligent ETL pipelines, RAG systems, and fine-tuning workflows using **LiquidAI** models and **Pydantic AI**.

Build secure, offline, compliant GenAI systems using local LiquidAI models — no cloud dependencies, no data exposure.

## 🎯 Features

- **Multi-Model Intelligence**: Uses different LiquidAI models for different tasks
  - LFM2-2.6B: Quality-focused chunking, metadata extraction, QA generation
  - LFM2-1.2B: Balanced summarization and rewriting
  - LFM2-700M: Fast validation and RAG inference
  
- **Auto Device Selection**: Automatically chooses CPU/GPU and optimal dtype based on available hardware

- **"Two Birds, One Stone" ETL**: While indexing documents, simultaneously generates fine-tuning datasets

- **Pydantic AI Integration**: Full type-safety, structured outputs, and graph-based orchestration

- **MCP Tools**: Model Context Protocol servers for RAG and filesystem operations

- **Docker Ready**: CPU and GPU containers with compose orchestration

## 🧭 Use Cases

| Use Case | Description |
| --- | --- |
| Secure RAG | No-cloud legal/medical knowledge bases |
| SOC Automation | Offline incident knowledge retrieval |
| Compliance AI | GDPR / ISO / NIST Q&A over policy docs |
| Edge AI | Deployed on laptops or secure enclaves |
| Multi-Agent Simulation | Autonomous incident response |

## 📈 Benchmarks & Hardware Expectations

| Model | Role | RAM | Speed CPU | Speed GPU |
| --- | --- | --- | --- | --- |
| 700M | RAG Runtime | ~4 GB | ⚡ Fast (15–22 tok/s) | ⚡⚡ Very Fast (90–120 tok/s) |
| 1.2B | Summaries | ~6 GB | ◼ Medium | ⚡⚡ Fast |
| 2.6B | Metadata / QA | ~10–12 GB | ◼ Slow | ⚡ Medium (≥12 GB VRAM recommended) |

## 🔭 Observability

- Built-in OpenTelemetry presence through Logfire
- Monitor response quality, token usage, and hallucination scores

```python
import logfire

logfire.configure()
logfire.instrument_pydantic_ai()
```

## 🛡️ Security Controls

This system follows security-by-design patterns aligned to ISO/IEC 27001 (confidentiality, integrity, logging), ISO/IEC 42010 architecture practices, and NIST SP 800-53 guardrails.

- Zero-trust network: containers can run with `--network none`
- Storage scoping: `/data` mount only
- Policy enforcement: no remote inference calls
- Audit traceability: agent tool calls are logged and typed

## 📦 Repository Structure

```
liquid-ai-stack/
├── liquid-shared-core/      # Shared utilities, models, schemas
├── liquid-etl-pipeline/     # Intelligent ETL with Pydantic AI graphs
├── liquid-rag-runtime/      # Fast RAG inference API
├── liquid-mcp-tools/        # MCP servers for tools
├── liquid-ft-trainer/       # Fine-tuning with LoRA/QLoRA
├── docker-compose.yml       # Full stack orchestration
└── README.md
```

## 📌 Current Implementation & Gaps

The stack ships with five core, production-ready components:

- **`liquid-shared-core`** – shared utilities, Pydantic models, and hardware helpers used by every service.
- **`liquid-etl-pipeline`** – multi-agent ETL that chunks documents, enriches metadata, summarizes, and prepares
  fine-tuning datasets alongside vector-store outputs.
- **`liquid-rag-runtime`** – FastAPI-based RAG API exposing health, full-agent `/ask`, simple `/ask/simple`, and
  `/search` endpoints with caching, metrics, and rate limits.
- **`liquid-mcp-tools`** – Model Context Protocol servers that expose filesystem and RAG helpers for agent
  interoperability.
- **`liquid-ft-trainer`** – LoRA/QLoRA fine-tuning utilities for LFM2 700M, 1.2B, and 2.6B models.

Remaining roadmap areas to implement:

- **Continuous evaluation loop** that routes low-confidence answers or user feedback back into automated
  fine-tuning jobs.
- **Policy/audit dashboards** to visualize access controls and compliance logs beyond the API responses.
- **User/admin UI** for chatting with agents and managing pipelines (the stack is currently API-centric/headless).
- **CI/CD and advanced testing** (no GitHub Actions or similar workflows are present yet).
- **Knowledge graph integration** to complement vector search with traversable semantic relationships.

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/liquid-ai-stack.git
cd liquid-ai-stack

# Create directories
mkdir -p models data/raw data/processed data/ft data/vectordb
```

### 2. Download Models

```bash
# Using Hugging Face CLI
pip install huggingface-hub

# Download LiquidAI models (choose based on your hardware)
huggingface-cli download LiquidAI/LFM2-700M --local-dir models/LFM2-700M
huggingface-cli download LiquidAI/LFM2-1.2B --local-dir models/LFM2-1.2B
huggingface-cli download LiquidAI/LFM2-2.6B --local-dir models/LFM2-2.6B
```

### 3. Install Dependencies

```bash
# Install shared core
cd liquid-shared-core
pip install -e .

# Install ETL pipeline
cd ../liquid-etl-pipeline
pip install -e .

# Install RAG runtime
cd ../liquid-rag-runtime
pip install -e .
```

### 4. Run ETL Pipeline

```bash
# Place documents in data/raw/
cp your-documents/*.pdf data/raw/

# Run ETL
cd liquid-etl-pipeline
python -m etl_pipeline.run_etl
```

### 5. Start RAG API

```bash
cd liquid-rag-runtime
python -m rag_runtime.api_server --port 8000
```

### 6. Query the API

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the key points about GDPR compliance?"}'
```

## 🐳 Docker Deployment

Deploy using our ready-to-run containers. Choose the profile that matches your hardware.

### CPU Mode

```bash
# Run full stack
docker-compose --profile cpu up

# Run ETL only
docker-compose run etl-cpu

# Run RAG API only
docker-compose --profile rag-cpu up
```

### GPU Mode

```bash
# Ensure NVIDIA Docker is installed
# Run with GPU support
docker-compose --profile gpu up

# Or specific services
docker-compose run etl
docker-compose --profile rag up
```

## 🏗️ Architecture

### System Context

```
+-------------+             +-----------------+
| User / SOC  |  HTTP API   | RAG Runtime     |
| Analyst     |────────────>| (700M Model)    |
+-------------+             +--------┬--------+
                                          │
                                          ▼
                                 +----------------+
                                 | Vector Store   |
                                 | (metadata)     |
                                 +----------------+
                                          │
                                    ETL Graph (2.6B + 1.2B + 700M)
                                          │
                                          ▼
                                  Fine-Tuning Dataset
```

### ETL Pipeline Graph

```
Raw Document
    │
    ▼
┌─────────────────┐
│ Chunking Agent  │ (LFM2-2.6B)
│ Semantic split  │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌─────────┐
│Metadata│ │ Summary │ (LFM2-1.2B)
│ Agent  │ │  Agent  │
└───┬────┘ └────┬────┘
    │           │
    └─────┬─────┘
          ▼
   ┌──────────────┐
   │  QA Agent    │ (LFM2-2.6B)
   │ Generate FT  │
   └──────┬───────┘
          ▼
   ┌──────────────┐
   │ Validation   │ (LFM2-700M)
   │    Agent     │
   └──────┬───────┘
          │
    ┌─────┴─────┐
    ▼           ▼
Vector DB    FT Dataset
(Chroma)      (JSONL)
```

### Agent Taxonomy

| Component | Model | Responsibility |
| --- | --- | --- |
| Chunking Agent | 2.6B | Semantic structure |
| Metadata Agent | 2.6B | Risk/entity extraction |
| Summary Agent | 1.2B | Knowledge condensation |
| QA Agent | 2.6B | FT dataset creation |
| Validation Agent | 700M | Hallucination filter |
| RAG Agent | 700M | Low-latency answers |

## 🧭 Governance & Maturity Roadmap

- Phase 1: Local intelligent ETL + RAG
- Phase 2: Agent-to-Agent MCP tools
- Phase 3: Continuous evaluation and retraining loop
- Phase 4: Policy-based access control + audit dashboards

### Device Selection Logic

```python
# Automatic based on hardware + model size
if no_cuda:
    device = "cpu", dtype = "fp32"
elif model_size == "2.6B" and gpu_mem >= 10GB:
    device = "cuda", dtype = "fp16"
elif model_size == "1.2B" and gpu_mem >= 6GB:
    device = "cuda", dtype = "bf16"
elif model_size == "700M" and gpu_mem >= 4GB:
    device = "cuda", dtype = "bf16"
else:
    device = "cpu", dtype = "fp32"
```

## 📚 API Reference

### RAG Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/ask` | POST | Ask a question (full agent) |
| `/ask/simple` | POST | Simple RAG (no agent overhead) |
| `/search` | POST | Search documents only |
| `/stats` | GET | Vector store statistics |

### Example Request

```python
import requests

response = requests.post(
    "http://localhost:8000/ask",
    json={
        "question": "What is GDPR?",
        "max_context_chunks": 5,
        "use_hybrid_search": True,
        "fast_mode": True
    }
)
print(response.json())
```

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CUDA_VISIBLE_DEVICES` | `0` | GPU device(s) to use. Empty for CPU |
| `HF_HOME` | `/models/.cache` | HuggingFace cache directory |
| `SANDBOX_ROOT` | `/data` | Root for filesystem operations |

### Fine-Tuning Config

```python
from ft_trainer import FTConfig

config = FTConfig(
    model_name="LiquidAI/LFM2-700M",
    use_lora=True,           # LoRA for efficiency
    lora_r=16,               # LoRA rank
    lora_alpha=32,           # LoRA alpha
    use_4bit=False,          # QLoRA for very large models
    num_epochs=3,
    batch_size=4,
    learning_rate=2e-5,
)
```

## 🔒 Security

- **Sandboxed Execution**: All file operations are restricted to `/data`
- **No Remote Code**: Models loaded with `trust_remote_code=False`
- **SafeTensors**: Uses safe serialization format
- **Network Isolation**: Docker containers can run with `--network none`

## 🙏 Acknowledgments

This project builds on **LiquidAI**, whose efficient, CPU-deployable open models enable secure offline inference.

## 📄 License

This project is licensed under the MIT License.

**Note**: LiquidAI models are under the LFM Open License v1.0:
- Free for personal and research use
- Commercial use allowed for companies <$10M revenue
- See [LiquidAI License](https://huggingface.co/LiquidAI/LFM2-1.2B) for details

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📚 Documentation

- [ETL Pipeline Guide](liquid-etl-pipeline/README.md)
- [RAG Runtime Guide](liquid-rag-runtime/README.md)
- [MCP Tools Guide](liquid-mcp-tools/README.md)
- [Fine-Tuning Guide](liquid-ft-trainer/README.md)
- [Pydantic AI Docs](https://ai.pydantic.dev/)
- [LiquidAI Models](https://huggingface.co/LiquidAI)
