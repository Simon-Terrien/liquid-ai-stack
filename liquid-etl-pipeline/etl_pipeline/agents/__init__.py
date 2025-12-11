# etl_pipeline/agents/__init__.py
"""
ETL Pipeline Agents using Pydantic AI + LiquidAI models.

Each agent uses an appropriately sized model per LFM2 recommendations:
- chunking_agent: LFM2-1.2B (optimal for data extraction tasks)
- metadata_agent: LFM2-1.2B (optimal for data extraction tasks)
- summary_agent: LFM2-700M (fast and efficient for summarization)
- qa_agent: LFM2-1.2B (optimal for data extraction tasks)
- validate_agent: LFM2-700M (fast quality filtering)

All models are specifically recommended by LFM2 docs for "agentic tasks,
data extraction, and RAG" use cases. Total VRAM usage: ~10.5GB < 16GB available.
"""

from .chunking_agent import (
    ChunkingOutput,
    chunk_document,
    chunk_document_sync,
    create_chunking_agent,
    get_chunking_agent,
)
from .metadata_agent import (
    MetadataOutput,
    create_metadata_agent,
    extract_metadata,
    extract_metadata_sync,
    get_metadata_agent,
)
from .qa_agent import (
    QAOutput,
    create_qa_agent,
    generate_qa_pairs,
    generate_qa_pairs_sync,
    get_qa_agent,
)
from .summary_agent import (
    SummaryOutput,
    create_summary_agent,
    generate_summaries,
    generate_summaries_sync,
    get_summary_agent,
)
from .validate_agent import (
    ValidationOutput,
    create_validate_agent,
    get_validate_agent,
    validate_qa_pairs,
    validate_qa_pairs_sync,
)

__all__ = [
    # Chunking
    "create_chunking_agent",
    "get_chunking_agent",
    "chunk_document",
    "chunk_document_sync",
    "ChunkingOutput",
    # Metadata
    "create_metadata_agent",
    "get_metadata_agent",
    "extract_metadata",
    "extract_metadata_sync",
    "MetadataOutput",
    # Summary
    "create_summary_agent",
    "get_summary_agent",
    "generate_summaries",
    "generate_summaries_sync",
    "SummaryOutput",
    # QA
    "create_qa_agent",
    "get_qa_agent",
    "generate_qa_pairs",
    "generate_qa_pairs_sync",
    "QAOutput",
    # Validation
    "create_validate_agent",
    "get_validate_agent",
    "validate_qa_pairs",
    "validate_qa_pairs_sync",
    "ValidationOutput",
]
