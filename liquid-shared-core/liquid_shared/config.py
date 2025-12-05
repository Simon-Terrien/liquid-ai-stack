# liquid_shared/config.py
"""
Configuration constants for the LiquidAI stack.
Centralized config for all repos in the multi-repo architecture.
"""
from pathlib import Path
from typing import Final

# Base directories
BASE_DIR: Final = Path(__file__).resolve().parent.parent

# Model paths - mounted into containers
MODELS_DIR: Final = Path("/models")   # Docker volume mount point
DATA_DIR: Final = BASE_DIR / "data"

# Default LiquidAI models (HuggingFace model IDs)
LFM_SMALL: Final = "LiquidAI/LFM2-700M"     # ~742M params - fast inference
LFM_MEDIUM: Final = "LiquidAI/LFM2-1.2B"    # ~1.17B params - balanced
LFM_LARGE: Final = "LiquidAI/LFM2-2.6B"     # ~2.57B params - quality focus

# Context window (same across all LFM2 models)
CONTEXT_TOKENS: Final = 32768

# Embedding model for RAG
EMBEDDING_MODEL: Final = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_DIM: Final = 768

# Chunking defaults
DEFAULT_CHUNK_SIZE: Final = 800      # characters
DEFAULT_CHUNK_OVERLAP: Final = 200   # characters

# RAG defaults
DEFAULT_TOP_K: Final = 5

# Generation defaults
DEFAULT_MAX_NEW_TOKENS: Final = 512
DEFAULT_TEMPERATURE: Final = 0.2

# Device memory thresholds (in GB) for automatic GPU selection
GPU_MEMORY_THRESHOLDS: Final = {
    "700m": 4,    # Minimum VRAM for 700M model
    "1.2b": 6,    # Minimum VRAM for 1.2B model
    "2.6b": 10,   # Minimum VRAM for 2.6B model
}

# Fine-tuning defaults
FT_BATCH_SIZE: Final = 4
FT_LEARNING_RATE: Final = 2e-5
FT_NUM_EPOCHS: Final = 3
FT_WARMUP_STEPS: Final = 100
