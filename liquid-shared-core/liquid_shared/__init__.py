# liquid_shared/__init__.py
"""
Liquid Shared Core - Common utilities for the LiquidAI stack.

This package provides:
- Configuration constants
- Device selection logic (CPU/GPU auto-detection)
- Local LiquidAI model wrapper for Pydantic AI
- Pydantic schemas for ETL, RAG, and fine-tuning
- Vector database wrapper
"""

from .config import (
    CONTEXT_TOKENS,
    DATA_DIR,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    LFM_LARGE,
    LFM_MEDIUM,
    LFM_SMALL,
    MODELS_DIR,
)
from .devices import (
    DeviceConfig,
    check_device_compatibility,
    recommend_device,
)
from .liquid_model import (
    LocalLiquidModel,
    create_balanced_model,
    create_fast_model,
    create_quality_model,
)
from .schemas import (
    AnswerOutput,
    # Chunking
    Chunk,
    ChunkingOutput,
    # Summaries
    ChunkSummary,
    DPOSample,
    # State
    ETLState,
    # Fine-tuning
    FineTuneSample,
    MetadataOutput,
    QAOutput,
    # QA
    QAPair,
    RAGContext,
    RAGState,
    # RAG
    RetrievalResult,
    SummaryOutput,
    ValidationOutput,
)
from .vectordb import (
    EmbeddingService,
    VectorStore,
)

__version__ = "0.1.0"

__all__ = [
    # Config
    "MODELS_DIR",
    "DATA_DIR",
    "LFM_SMALL",
    "LFM_MEDIUM",
    "LFM_LARGE",
    "CONTEXT_TOKENS",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIM",
    # Devices
    "recommend_device",
    "check_device_compatibility",
    "DeviceConfig",
    # Models
    "LocalLiquidModel",
    "create_quality_model",
    "create_balanced_model",
    "create_fast_model",
    # Schemas
    "Chunk",
    "ChunkingOutput",
    "MetadataOutput",
    "ChunkSummary",
    "SummaryOutput",
    "QAPair",
    "QAOutput",
    "ValidationOutput",
    "FineTuneSample",
    "DPOSample",
    "RetrievalResult",
    "RAGContext",
    "AnswerOutput",
    "ETLState",
    "RAGState",
    # Vector DB
    "VectorStore",
    "EmbeddingService",
]
