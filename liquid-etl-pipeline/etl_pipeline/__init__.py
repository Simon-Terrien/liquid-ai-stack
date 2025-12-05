# etl_pipeline/__init__.py
"""
Liquid ETL Pipeline - Intelligent document processing with LiquidAI + Pydantic AI.

Features:
- Semantic chunking using LFM2-2.6B
- Rich metadata extraction
- Embedding-optimized summaries
- QA pair generation for fine-tuning
- Quality validation and filtering
- Vector store indexing
- Fine-tuning dataset generation

Usage:
    from etl_pipeline import run_etl
    
    stats = run_etl(input_dir=Path("./documents"))
"""

from .agents import (
    chunk_document,
    chunk_document_sync,
    # Chunking
    create_chunking_agent,
    # Metadata
    create_metadata_agent,
    # QA
    create_qa_agent,
    # Summary
    create_summary_agent,
    # Validation
    create_validate_agent,
    extract_metadata,
    extract_metadata_sync,
    generate_qa_pairs,
    generate_qa_pairs_sync,
    generate_summaries,
    generate_summaries_sync,
    validate_qa_pairs,
    validate_qa_pairs_sync,
)
from .graph_etl import (
    ETLOutput,
    ETLState,
    build_etl_graph,
    get_etl_graph,
    run_etl_pipeline,
    run_etl_pipeline_sync,
)
from .run_etl import (
    process_document,
    run_etl,
)

__version__ = "0.1.0"

__all__ = [
    # Graph
    "build_etl_graph",
    "get_etl_graph",
    "run_etl_pipeline",
    "run_etl_pipeline_sync",
    "ETLState",
    "ETLOutput",
    # Runner
    "run_etl",
    "process_document",
    # Agents
    "create_chunking_agent",
    "chunk_document",
    "chunk_document_sync",
    "create_metadata_agent",
    "extract_metadata",
    "extract_metadata_sync",
    "create_summary_agent",
    "generate_summaries",
    "generate_summaries_sync",
    "create_qa_agent",
    "generate_qa_pairs",
    "generate_qa_pairs_sync",
    "create_validate_agent",
    "validate_qa_pairs",
    "validate_qa_pairs_sync",
]
