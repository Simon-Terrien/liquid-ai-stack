# liquid_shared/schemas.py
"""
Pydantic schemas for the LiquidAI stack.

Defines structured types for:
- Document chunks and metadata
- Summaries and embeddings
- QA pairs for fine-tuning
- Fine-tuning dataset samples
"""
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ImportanceLevel(int, Enum):
    """Importance level for chunks (0-10 scale)."""
    VERY_LOW = 0
    LOW = 2
    BELOW_AVERAGE = 4
    AVERAGE = 5
    ABOVE_AVERAGE = 6
    HIGH = 8
    CRITICAL = 10


# ============================================================================
# Chunking & Metadata Schemas
# ============================================================================

class Chunk(BaseModel):
    """
    A document chunk with extracted metadata.
    
    Used during ETL to store processed text segments.
    """
    id: str = Field(description="Unique chunk identifier")
    text: str = Field(description="The chunk text content")
    section_title: str | None = Field(
        default=None,
        description="Extracted section/heading title"
    )
    source_path: str = Field(description="Path to source document")
    chunk_index: int = Field(ge=0, description="Position in source document")
    importance: int = Field(
        ge=0, le=10, default=5,
        description="Importance score (0=low, 10=critical)"
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Topic tags extracted from content"
    )
    entities: list[str] = Field(
        default_factory=list,
        description="Named entities (orgs, laws, systems, etc.)"
    )

    @field_validator("importance", mode="before")
    @classmethod
    def clamp_importance(cls, v: Any) -> int:
        """Ensure importance is within valid range."""
        if isinstance(v, (int, float)):
            return max(0, min(10, int(v)))
        return 5


class ChunkingOutput(BaseModel):
    """Output from the chunking agent."""
    chunks: list[str] = Field(description="List of chunk texts")


class MetadataOutput(BaseModel):
    """Output from the metadata extraction agent."""
    chunks: list[Chunk] = Field(description="Chunks with extracted metadata")


# ============================================================================
# Summary & Embedding Schemas
# ============================================================================

class ChunkSummary(BaseModel):
    """
    Summary and embedding-optimized text for a chunk.
    
    The embedding_text is specifically optimized for vector retrieval,
    while the summary is human-readable.
    """
    chunk_id: str = Field(description="Reference to parent chunk ID")
    summary: str = Field(description="Human-readable 2-4 sentence summary")
    embedding_text: str = Field(
        description="Embedding-optimized text (factual, dense, key entities)"
    )


class SummaryOutput(BaseModel):
    """Output from the summary agent."""
    summaries: list[ChunkSummary] = Field(description="Summaries for all chunks")


# ============================================================================
# QA Generation Schemas
# ============================================================================

class QAPair(BaseModel):
    """
    A question-answer pair for fine-tuning.
    
    Generated from chunk content for supervised learning.
    """
    chunk_id: str = Field(description="Reference to source chunk ID")
    question: str = Field(description="Realistic user query")
    answer: str = Field(description="Grounded answer from chunk content")
    quality_score: float | None = Field(
        default=None, ge=0, le=1,
        description="Quality score from validation (0-1)"
    )


class QAOutput(BaseModel):
    """Output from the QA generation agent."""
    qa_pairs: list[QAPair] = Field(description="Generated QA pairs")


class ValidationOutput(BaseModel):
    """Output from the QA validation agent."""
    validated_pairs: list[QAPair] = Field(
        description="Filtered high-quality QA pairs"
    )


# ============================================================================
# Fine-Tuning Dataset Schemas
# ============================================================================

class FineTuneSample(BaseModel):
    """
    A single sample for fine-tuning in instruction format.
    
    Compatible with common fine-tuning frameworks (HuggingFace, Axolotl).
    """
    instruction: str = Field(description="The instruction/question")
    input: str = Field(
        default="",
        description="Optional context/input (e.g., retrieved chunks)"
    )
    output: str = Field(description="The expected response")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (chunk_id, source, etc.)"
    )


class DPOSample(BaseModel):
    """
    A sample for Direct Preference Optimization (DPO) training.
    """
    prompt: str = Field(description="The prompt/question")
    chosen: str = Field(description="Preferred response")
    rejected: str = Field(description="Non-preferred response")
    metadata: dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# RAG Runtime Schemas
# ============================================================================

class RetrievalResult(BaseModel):
    """Result from vector retrieval."""
    chunk_id: str
    text: str
    score: float = Field(ge=0, le=1, description="Similarity score")
    metadata: dict[str, Any] = Field(default_factory=dict)


class RAGContext(BaseModel):
    """Context assembled for RAG generation."""
    query: str
    retrieved_chunks: list[RetrievalResult]
    total_tokens: int | None = None


class AnswerOutput(BaseModel):
    """Output from the RAG answer agent."""
    answer: str = Field(description="Generated answer")
    used_chunk_ids: list[str] = Field(
        default_factory=list,
        description="IDs of chunks used in answer"
    )
    confidence: float | None = Field(
        default=None, ge=0, le=1,
        description="Confidence score"
    )


# ============================================================================
# Pipeline State Schemas
# ============================================================================

class ETLState(BaseModel):
    """
    State object for the ETL pipeline graph.
    
    Tracks progress through chunking -> metadata -> summary -> QA stages.
    """
    source_path: str
    raw_text: str = ""
    chunks: list[Chunk] = Field(default_factory=list)
    summaries: list[ChunkSummary] = Field(default_factory=list)
    qa_pairs: list[QAPair] = Field(default_factory=list)
    ft_samples: list[FineTuneSample] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    processed_at: datetime | None = None


class RAGState(BaseModel):
    """
    State object for the RAG pipeline graph.
    """
    query: str
    retrieved_chunks: list[RetrievalResult] = Field(default_factory=list)
    reranked_chunks: list[RetrievalResult] = Field(default_factory=list)
    answer: str | None = None
    sources: list[str] = Field(default_factory=list)
