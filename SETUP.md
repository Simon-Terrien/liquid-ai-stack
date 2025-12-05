# Setup Guide - LiquidAI Stack

This guide walks you through setting up the LiquidAI Stack with UV package manager.

## Prerequisites

- Python 3.10 or higher
- UV package manager (already installed: v0.9.9)
- 20GB+ free disk space (for all models)
- Optional: NVIDIA GPU with 4-10GB VRAM

## Quick Start (Recommended)

### 1. Initialize UV Environment and Install Dependencies

```bash
# Create and sync UV environment with all dependencies
uv sync

# This will:
# - Create a virtual environment in .venv/
# - Install all dependencies from pyproject.toml
# - Install all local packages (liquid-shared-core, liquid-etl-pipeline, etc.) in editable mode
```

### 2. Download LiquidAI Models

```bash
# Option A: Download all models (700M, 1.2B, 2.6B) - ~18GB total
uv run bootstrap.py

# Option B: Download only the small model (700M) - ~3GB
uv run bootstrap.py --model 700M

# Option C: Download medium model (1.2B) - ~5GB
uv run bootstrap.py --model 1.2B

# Option D: Download large model (2.6B) - ~10GB
uv run bootstrap.py --model 2.6B

# Option E: Skip model downloads (for environment check only)
uv run bootstrap.py --skip-models

# Option F: List available models
uv run bootstrap.py --list
```

### 3. Verify Installation

```bash
# Check environment and dependencies
uv run bootstrap.py --skip-models

# Test imports
uv run python -c "from liquid_shared import config, devices, schemas; print('✓ Imports successful')"

# Check device compatibility
uv run python -c "from liquid_shared.devices import check_device_compatibility; import json; print(json.dumps(check_device_compatibility(), indent=2))"
```

## Manual Setup (Alternative)

If you prefer not to use UV:

### 1. Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install Dependencies

```bash
# Install in order (shared-core must be first)
pip install -e ./liquid-shared-core
pip install -e ./liquid-etl-pipeline
pip install -e ./liquid-rag-runtime
pip install -e ./liquid-mcp-tools
pip install -e ./liquid-ft-trainer
```

### 3. Download Models

```bash
# Install HuggingFace CLI if not already installed
pip install huggingface-hub

# Run bootstrap script
python bootstrap.py
```

## Running the Stack

### ETL Pipeline

Process documents from `data/raw/` into vector embeddings and fine-tuning datasets:

```bash
# Using UV
uv run liquid-etl

# Or directly
uv run python -m etl_pipeline.run_etl
```

### RAG API Server

Start the FastAPI server for question answering:

```bash
# Using UV
uv run liquid-rag-server --port 8000

# Or directly
uv run python -m rag_runtime.api_server --port 8000
```

Then query it:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is in the documents?", "fast_mode": true}'
```

### Docker Deployment

```bash
# GPU mode (requires nvidia-docker)
docker-compose --profile gpu up

# CPU mode
docker-compose --profile cpu up

# Specific services
docker-compose run etl              # ETL pipeline (GPU)
docker-compose run etl-cpu          # ETL pipeline (CPU)
docker-compose --profile rag up     # RAG API (GPU)
docker-compose --profile rag-cpu up # RAG API (CPU)
```

## Model Download Details

The bootstrap script downloads models to `./models/`:

| Model | Size | Use Case | RAM Required | Disk Space |
|-------|------|----------|--------------|------------|
| LFM2-700M | ~742M params | Fast inference, validation, RAG | 4GB | ~3GB |
| LFM2-1.2B | ~1.17B params | Summarization, rewriting | 6GB | ~5GB |
| LFM2-2.6B | ~2.57B params | Chunking, metadata, QA generation | 10GB | ~10GB |

**Download options:**
- `--model 700M`: Download only 700M model (fastest, CPU-friendly)
- `--model 1.2B`: Download only 1.2B model (balanced)
- `--model 2.6B`: Download only 2.6B model (highest quality)
- `--model all`: Download all models (default, recommended)
- `--force`: Force re-download even if models exist
- `--skip-models`: Skip downloads, only check environment

## Directory Structure

After setup, you should have:

```
liquid-ai-stack/
├── .venv/                   # UV virtual environment
├── models/                  # Downloaded LiquidAI models
│   ├── LFM2-700M/
│   ├── LFM2-1.2B/
│   └── LFM2-2.6B/
├── data/
│   ├── raw/                 # Input documents (PDF, TXT, etc.)
│   ├── processed/           # ETL outputs
│   ├── vectordb/            # Chroma vector database
│   └── ft/                  # Fine-tuning datasets
├── liquid-shared-core/
├── liquid-etl-pipeline/
├── liquid-rag-runtime/
├── liquid-mcp-tools/
├── liquid-ft-trainer/
├── bootstrap.py             # Model download script
├── pyproject.toml           # Root project config
└── README.md
```

## Troubleshooting

### UV Environment Issues

```bash
# Remove and recreate environment
rm -rf .venv
uv sync

# Check UV version
uv --version

# Update UV
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Model Download Failures

```bash
# Check HuggingFace Hub
uv run python -c "import huggingface_hub; print(huggingface_hub.__version__)"

# Re-download with force flag
uv run bootstrap.py --force --model 700M

# Check disk space
df -h .
```

### Import Errors

```bash
# Verify all packages are installed
uv pip list | grep liquid

# Reinstall in editable mode
uv sync --reinstall
```

### GPU/CUDA Issues

```bash
# Check CUDA availability
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check device recommendations
uv run python liquid-shared-core/liquid_shared/devices.py
```

### Docker Issues

```bash
# Rebuild images
docker-compose build --no-cache

# Check logs
docker-compose logs etl
docker-compose logs rag

# Clean up
docker-compose down -v
```

## Development Workflow

### Running Tests

```bash
# Run all tests
uv run pytest

# Run specific package tests
uv run pytest liquid-etl-pipeline/tests/
uv run pytest liquid-rag-runtime/tests/

# Run with coverage
uv run pytest --cov=liquid_shared --cov=etl_pipeline --cov=rag_runtime
```

### Code Formatting

```bash
# Format code
uv run black .

# Check with ruff
uv run ruff check .

# Auto-fix issues
uv run ruff check --fix .

# Type checking
uv run mypy liquid-shared-core/liquid_shared
```

### Adding Dependencies

```bash
# Add to root project
uv add <package-name>

# Add to specific sub-package
cd liquid-etl-pipeline
uv add <package-name>
```

## GPU Recommendations

Based on your system (32 CPUs, 19.5GB RAM):

- **For CPU-only**: Start with 700M model
- **For GPU (4-6GB VRAM)**: Use 700M or 1.2B models
- **For GPU (8-10GB VRAM)**: Use 1.2B or 2.6B models
- **For GPU (12GB+ VRAM)**: Use all models, run 2.6B for quality tasks

## Next Steps

1. ✅ Install dependencies: `uv sync`
2. ✅ Download models: `uv run bootstrap.py`
3. 📄 Add documents to `data/raw/`
4. 🔄 Run ETL: `uv run liquid-etl`
5. 🚀 Start RAG API: `uv run liquid-rag-server`
6. 💬 Query your documents!

## Resources

- [UV Documentation](https://docs.astral.sh/uv/)
- [LiquidAI Models](https://huggingface.co/LiquidAI)
- [Pydantic AI](https://ai.pydantic.dev/)
- [Project README](README.md)
- [Claude Code Guide](CLAUDE.md)
