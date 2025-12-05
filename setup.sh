#!/bin/bash
# setup.sh - Setup script for LiquidAI Multi-Agent Stack
# Usage: ./setup.sh [--cpu-only] [--download-models]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse arguments
CPU_ONLY=false
DOWNLOAD_MODELS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --cpu-only)
            CPU_ONLY=true
            shift
            ;;
        --download-models)
            DOWNLOAD_MODELS=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--cpu-only] [--download-models]"
            echo ""
            echo "Options:"
            echo "  --cpu-only        Skip GPU-specific setup"
            echo "  --download-models Download LiquidAI models"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "LiquidAI Multi-Agent Stack Setup"
echo "=========================================="

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
if [[ $(echo "$PYTHON_VERSION < 3.10" | bc -l) -eq 1 ]]; then
    echo "Error: Python 3.10+ required (found $PYTHON_VERSION)"
    exit 1
fi
echo "✓ Python version: $PYTHON_VERSION"

# Create directories
echo "Creating directories..."
mkdir -p models data/raw data/processed data/ft data/vectordb
echo "✓ Directories created"

# Create virtual environment (optional)
if [[ ! -d "venv" ]]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
else
    source venv/bin/activate
fi
echo "✓ Virtual environment ready"

# Install shared core
echo "Installing liquid-shared-core..."
cd liquid-shared-core
pip install -e . --quiet
cd ..
echo "✓ liquid-shared-core installed"

# Install ETL pipeline
echo "Installing liquid-etl-pipeline..."
cd liquid-etl-pipeline
pip install -e . --quiet
cd ..
echo "✓ liquid-etl-pipeline installed"

# Install RAG runtime
echo "Installing liquid-rag-runtime..."
cd liquid-rag-runtime
pip install -e . --quiet
cd ..
echo "✓ liquid-rag-runtime installed"

# Install MCP tools
echo "Installing liquid-mcp-tools..."
cd liquid-mcp-tools
pip install -e . --quiet
cd ..
echo "✓ liquid-mcp-tools installed"

# Install FT trainer
echo "Installing liquid-ft-trainer..."
cd liquid-ft-trainer
pip install -e . --quiet
cd ..
echo "✓ liquid-ft-trainer installed"

# Check CUDA
if [[ "$CPU_ONLY" == false ]]; then
    if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
        echo "✓ CUDA available"
        GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
        GPU_MEM=$(python3 -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')")
        echo "  GPU: $GPU_NAME ($GPU_MEM)"
    else
        echo "⚠ CUDA not available - will use CPU"
    fi
fi

# Download models
if [[ "$DOWNLOAD_MODELS" == true ]]; then
    echo ""
    echo "Downloading LiquidAI models..."
    pip install huggingface-hub --quiet
    
    echo "Downloading LFM2-700M (recommended for CPU)..."
    huggingface-cli download LiquidAI/LFM2-700M --local-dir models/LFM2-700M
    
    echo "Downloading LFM2-1.2B (balanced)..."
    huggingface-cli download LiquidAI/LFM2-1.2B --local-dir models/LFM2-1.2B
    
    if [[ "$CPU_ONLY" == false ]]; then
        echo "Downloading LFM2-2.6B (quality-focused)..."
        huggingface-cli download LiquidAI/LFM2-2.6B --local-dir models/LFM2-2.6B
    fi
    
    echo "✓ Models downloaded"
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Quick Start:"
echo ""
echo "1. Place documents in data/raw/"
echo "   cp your-documents/*.pdf data/raw/"
echo ""
echo "2. Run ETL pipeline:"
echo "   source venv/bin/activate"
echo "   python -m etl_pipeline.run_etl"
echo ""
echo "3. Start RAG API:"
echo "   python -m rag_runtime.api_server --port 8000"
echo ""
echo "4. Query the API:"
echo "   curl -X POST http://localhost:8000/ask \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"question\": \"Your question here\"}'"
echo ""

if [[ "$DOWNLOAD_MODELS" == false ]]; then
    echo "Note: Run with --download-models to download LiquidAI models"
fi
