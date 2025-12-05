# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a multi-repository Python monorepo for building production-ready ETL pipelines, RAG systems, and fine-tuning workflows using **LiquidAI models** (LFM2 series) and **Pydantic AI**. The architecture emphasizes type safety, structured outputs, and intelligent multi-model orchestration.

**Key Philosophy**: "Two Birds, One Stone" ETL - while indexing documents for RAG, simultaneously generate fine-tuning datasets.

## Repository Structure

```
liquid-ai-stack/
├── liquid-shared-core/      # Shared utilities, models, schemas, device selection
├── liquid-etl-pipeline/     # ETL with Pydantic Graph orchestration
├── liquid-rag-runtime/      # FastAPI RAG inference server
├── liquid-mcp-tools/        # Model Context Protocol servers
├── liquid-ft-trainer/       # Fine-tuning with LoRA/QLoRA
└── docker-compose.yml       # Full stack orchestration
```

Each component is a separate Python package with its own `pyproject.toml`.

## Common Commands

### Setup & Installation

```bash
# Install all components (run from each subdirectory)
cd liquid-shared-core && pip install -e .
cd ../liquid-etl-pipeline && pip install -e .
cd ../liquid-rag-runtime && pip install -e .
cd ../liquid-mcp-tools && pip install -e .
cd ../liquid-ft-trainer && pip install -e .

# Download LiquidAI models (if not already present)
huggingface-cli download LiquidAI/LFM2-700M --local-dir models/LFM2-700M
huggingface-cli download LiquidAI/LFM2-1.2B --local-dir models/LFM2-1.2B
huggingface-cli download LiquidAI/LFM2-2.6B --local-dir models/LFM2-2.6B
```

### Running Components

```bash
# Run ETL pipeline (processes documents in data/raw/)
cd liquid-etl-pipeline
python -m etl_pipeline.run_etl
# Or using the CLI entry point:
liquid-etl

# Run RAG API server
cd liquid-rag-runtime
python -m rag_runtime.api_server --port 8000
# Or:
liquid-rag-server --port 8000

# Run Fine-tuning
cd liquid-ft-trainer
python -m ft_trainer.train --lora
```

### Docker Operations

```bash
# GPU mode (requires nvidia-docker)
docker-compose --profile gpu up          # Full stack
docker-compose run etl                   # ETL only
docker-compose --profile rag up          # RAG API only
docker-compose --profile train up        # Fine-tuning only

# CPU mode
docker-compose --profile cpu up          # Full stack
docker-compose run etl-cpu               # ETL only
docker-compose --profile rag-cpu up      # RAG API only
```

### Testing & Linting

```bash
# Each component uses the same testing setup
cd <component>
pytest                     # Run all tests
pytest tests/test_foo.py   # Run specific test file
pytest -k "test_name"      # Run tests matching pattern

# Linting (configured in pyproject.toml)
ruff check .              # Check for issues
ruff check --fix .        # Auto-fix issues
black .                   # Format code

# Type checking
mypy .
```

## Architecture Details

### Multi-Model Strategy

The stack uses **different LiquidAI models for different tasks** based on the quality-speed tradeoff:

- **LFM2-2.6B**: Quality-focused tasks (chunking, metadata extraction, QA generation)
- **LFM2-1.2B**: Balanced tasks (summarization, rewriting)
- **LFM2-700M**: Fast tasks (validation, RAG inference)

All models share the same 32K token context window.

### Automatic Device Selection

**Critical**: The `liquid_shared.devices` module automatically selects CPU vs GPU and optimal dtype based on:
- CUDA availability
- GPU VRAM (via `torch.cuda.get_device_properties`)
- Model size (inferred from model name patterns like "2.6B", "1.2B", "700M")

**Logic** (see `liquid-shared-core/liquid_shared/devices.py:68`):
- Large models (2.6B): Require 10GB+ VRAM for GPU, use fp16
- Medium models (1.2B): Require 6GB+ VRAM, prefer bf16 if supported
- Small models (700M): Require 4GB+ VRAM, prefer bf16
- Fallback to CPU with fp32 if insufficient VRAM

Always use `recommend_device(model_name)` instead of hardcoding device selection.

### ETL Pipeline Graph

The ETL pipeline uses **Pydantic Graph** (beta API) for orchestration. The graph is defined in `liquid-etl-pipeline/etl_pipeline/graph_etl.py:75`.

**Flow**:
1. **Chunking Agent** → Semantic document splitting (LFM2-2.6B)
2. **Metadata Agent** → Extract section titles, tags, entities, importance (LFM2-2.6B)
3. **Summary Agent** → Generate summaries and embedding-optimized text (LFM2-1.2B)
4. **QA Agent** → Generate question-answer pairs for fine-tuning (LFM2-2.6B)
5. **Validation Agent** → Filter low-quality QA pairs (LFM2-700M)
6. **Output**: Vector store (Chroma) + Fine-tuning dataset (JSONL)

**Key Implementation Details**:
- Graph state is managed via `ETLState` dataclass (`graph_etl.py:40`)
- Steps are async but agents are sync (use `chunk_document_sync`, etc.)
- Metadata and summary steps run in parallel conceptually, but the graph implementation executes them sequentially
- Each step updates the shared state object
- Errors are collected in `state.errors` rather than failing fast

### Shared Schemas

All components use **Pydantic schemas** from `liquid-shared-core/liquid_shared/schemas.py`. Key types:

- **`Chunk`**: Document chunk with metadata (section_title, tags, entities, importance 0-10)
- **`ChunkSummary`**: Human-readable summary + embedding-optimized text
- **`QAPair`**: Question-answer pair with quality_score
- **`FineTuneSample`**: Instruction format for fine-tuning (instruction, input, output, metadata)
- **`RetrievalResult`**: Vector search result (chunk_id, text, score, metadata)

These schemas ensure type safety across all pipeline stages.

### RAG Runtime

The RAG runtime (`liquid-rag-runtime/`) is a FastAPI server with three modes:

1. **Full Agent** (`/ask`): Uses Pydantic AI agent with retrieval tools (slower, more sophisticated)
2. **Simple RAG** (`/ask/simple`): Direct retrieval + generation (faster, no agent overhead)
3. **Search Only** (`/search`): Returns relevant chunks without generation

**Key Files**:
- `api_server.py:117`: `/ask` endpoint with full agent
- `api_server.py:154`: `/ask/simple` endpoint for fast inference
- `rag_agent.py`: Agent definition with retrieval tools
- `tools/retrieval.py`: Vector search and hybrid search implementations

**Performance Tip**: Use `fast_mode=True` (default) to use LFM2-700M instead of 1.2B for inference.

### Configuration System

Central configuration is in `liquid-shared-core/liquid_shared/config.py`:

- **Model paths**: Expects models in `/models/` (Docker mount point)
- **Data paths**: `/data/` for raw docs, processed data, vector DB
- **Model IDs**: `LFM_SMALL`, `LFM_MEDIUM`, `LFM_LARGE` constants
- **Embedding model**: `sentence-transformers/all-mpnet-base-v2` (768-dim)
- **Chunking defaults**: 800 chars with 200 char overlap
- **GPU thresholds**: Memory requirements for each model size

When modifying defaults, update `config.py` rather than hardcoding values.

### MCP Tools

The `liquid-mcp-tools/` package provides Model Context Protocol servers:

- **RAG Server**: Exposes RAG capabilities as MCP tools
- **Filesystem Server**: Sandboxed file operations (restricted to `/data/`)

These enable Pydantic AI agents to call RAG operations and file operations via standardized MCP interface.

### Docker & Environment

**Environment Variables**:
- `CUDA_VISIBLE_DEVICES`: GPU device selection (empty string = CPU mode)
- `HF_HOME`: HuggingFace cache directory (default: `/models/.cache`)
- `PYTHONPATH`: Must include `/app` and `/app/liquid-shared-core` in containers

**Profiles**:
- `gpu`: GPU-accelerated services (requires nvidia-docker)
- `cpu`: CPU-only services
- `etl`, `rag`, `train`: Individual service profiles
- Combine profiles: `docker-compose --profile gpu --profile rag up`

**Volume Mounts**:
- `./models:/models:ro` (read-only, contains downloaded LiquidAI models)
- `./data:/data` (read-write, contains raw docs, processed data, vector DB)
- `./liquid-shared-core:/app/liquid-shared-core:ro` (shared code)

## Development Guidelines

### When Adding New Agents

1. Define input/output schemas in `liquid-shared-core/liquid_shared/schemas.py`
2. Implement agent in appropriate package (e.g., `etl_pipeline/agents/`)
3. Use `recommend_device()` for device selection
4. Return structured Pydantic models, not raw strings
5. Update the graph if adding to ETL pipeline (`graph_etl.py`)

### When Modifying the ETL Graph

- The graph is defined in `build_etl_graph()` at `graph_etl.py:75`
- Steps are decorated with `@g.step`
- Use `StepContext[ETLState, None, InputType]` for type safety
- Update state via `ctx.state.field_name = value`
- Wire edges with `g.edge_from(source).to(target)`
- Test with `graph.render()` to visualize changes

### When Adding API Endpoints

- Add request/response models as Pydantic BaseModel subclasses
- Use FastAPI dependency injection for shared resources
- Add proper error handling with HTTPException
- Log operations with structured logging (`logger.info(f"...")`)
- Update OpenAPI docs via docstrings

### Code Style

- Line length: 100 characters (enforced by ruff)
- Python 3.10+ (uses match statements, type union syntax)
- Strict mypy mode enabled (ignore_missing_imports=True)
- Use async def for graph steps, sync functions for agent implementations
- Type all function signatures with proper return types

### Security Considerations

- All file operations in MCP tools are sandboxed to `/data/`
- Models loaded with `trust_remote_code=False`
- SafeTensors format used for model weights
- CORS enabled on API server (restrict in production)
- No secrets in environment variables (use volume mounts for sensitive configs)

## Common Patterns

### Loading a LiquidAI Model

```python
from liquid_shared import recommend_device, LFM_SMALL
from transformers import AutoModelForCausalLM, AutoTokenizer

# Automatic device selection
config = recommend_device(LFM_SMALL)
print(f"Using: {config.device} with {config.dtype_name}")

model = AutoModelForCausalLM.from_pretrained(
    LFM_SMALL,
    torch_dtype=config.torch_dtype,
    device_map=config.device,
    trust_remote_code=False,
)
tokenizer = AutoTokenizer.from_pretrained(LFM_SMALL)
```

### Creating a Pydantic AI Agent

```python
from pydantic_ai import Agent
from liquid_shared import ChunkingOutput

agent = Agent(
    "outlines-transformers:LiquidAI/LFM2-2.6B",
    result_type=ChunkingOutput,
    system_prompt="You are a document chunking assistant...",
)

result = agent.run_sync("Chunk this document: ...")
chunks = result.data.chunks
```

### Adding a New Graph Step

```python
@g.step
async def new_step(ctx: StepContext[ETLState, None, List[Chunk]]) -> OutputType:
    """Description of what this step does."""
    try:
        # Process ctx.inputs
        result = process_data(ctx.inputs)

        # Update state
        ctx.state.new_field = result

        return result
    except Exception as e:
        ctx.state.errors.append(f"Step failed: {e}")
        raise

# Wire it up
g.edge_from(previous_step).to(new_step)
```

### Vector Store Operations

```python
from liquid_shared import VectorStore, EmbeddingService

# Initialize
embedding_service = EmbeddingService()
vector_store = VectorStore(
    collection_name="documents",
    embedding_service=embedding_service,
)

# Add chunks
vector_store.add_chunks(chunks, summaries)

# Search
results = vector_store.search(
    query="What is GDPR?",
    top_k=5,
)
```

## Troubleshooting

**Problem**: Models fail to load with CUDA OOM
- **Solution**: Check device selection with `check_device_compatibility()` from `liquid_shared.devices`
- **Solution**: Set `CUDA_VISIBLE_DEVICES=""` to force CPU mode
- **Solution**: Use smaller model variant (700M instead of 2.6B)

**Problem**: ETL pipeline hangs at a specific step
- **Solution**: Check `state.errors` for accumulated errors
- **Solution**: Enable debug logging: `logging.basicConfig(level=logging.DEBUG)`
- **Solution**: Run step functions directly outside graph for debugging

**Problem**: Vector store empty after ETL
- **Solution**: Verify `data/vectordb/` has Chroma database files
- **Solution**: Check ETL output stats: `result.stats` should show `num_chunks > 0`
- **Solution**: Ensure embeddings were generated: check `summaries` list

**Problem**: Docker container exits immediately
- **Solution**: Check logs: `docker logs <container_name>`
- **Solution**: Verify model files exist in `./models/`
- **Solution**: Check PYTHONPATH includes liquid-shared-core

## File Locations Reference

**Configuration**: `liquid-shared-core/liquid_shared/config.py`
**Device Selection**: `liquid-shared-core/liquid_shared/devices.py:68`
**Schemas**: `liquid-shared-core/liquid_shared/schemas.py`
**ETL Graph**: `liquid-etl-pipeline/etl_pipeline/graph_etl.py:75`
**ETL Entry Point**: `liquid-etl-pipeline/etl_pipeline/run_etl.py`
**RAG API**: `liquid-rag-runtime/rag_runtime/api_server.py`
**RAG Agent**: `liquid-rag-runtime/rag_runtime/rag_agent.py`
**Vector Store**: `liquid-shared-core/liquid_shared/vectordb.py`
**Docker Compose**: `docker-compose.yml`
