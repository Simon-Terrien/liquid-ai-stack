# etl_pipeline/agents/chunking_agent.py
"""
Intelligent chunking agent using LiquidAI 2.6B.

Splits documents into semantically coherent chunks based on:
- Topic transitions
- Section boundaries
- Paragraph structure
- Logical coherence
"""

from liquid_shared import LFM_LARGE, LocalLiquidModel
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings


class ChunkBoundary(BaseModel):
    """Represents a chunk boundary in the document."""
    start_char: int = Field(description="Start character position")
    end_char: int = Field(description="End character position")
    reason: str = Field(description="Why this boundary was chosen")


class ChunkingOutput(BaseModel):
    """Output from the intelligent chunking process."""
    chunks: list[str] = Field(description="List of chunk texts")
    boundaries: list[ChunkBoundary] = Field(
        default_factory=list,
        description="Boundary information for each chunk"
    )


CHUNKING_INSTRUCTIONS = """You are an expert document analyzer specializing in semantic text segmentation.

Your task is to split the provided document into semantically coherent chunks. Each chunk should:

1. **Be Self-Contained**: A chunk should make sense on its own, containing a complete thought or topic.

2. **Respect Natural Boundaries**:
   - Section/heading transitions
   - Topic shifts
   - Paragraph breaks that signal new ideas
   - List completions
   - Definition completions

3. **Optimal Size**: Aim for 500-1500 tokens per chunk. Smaller is acceptable for distinct sections.

4. **Preserve Context**: Never split:
   - Mid-sentence
   - In the middle of a list
   - Between a term and its definition
   - Between a heading and its content

5. **Handle Special Cases**:
   - Tables: Keep complete or split by logical row groups
   - Code blocks: Keep complete
   - Nested lists: Keep hierarchy intact

CRITICAL: Return valid JSON with a list of chunk texts. Each chunk text should be the exact text from the document, preserving all formatting and whitespace.

Example output structure:
{
  "chunks": [
    "First chunk text here...",
    "Second chunk text here...",
    "Third chunk text here..."
  ],
  "boundaries": [
    {"start_char": 0, "end_char": 500, "reason": "Introduction section"},
    {"start_char": 500, "end_char": 1200, "reason": "Topic shift to methodology"}
  ]
}
"""


def create_chunking_agent(model: LocalLiquidModel = None) -> Agent:
    """
    Create the chunking agent.
    
    Args:
        model: Optional pre-loaded model. If None, loads LFM2-2.6B.
        
    Returns:
        Configured Pydantic AI Agent
    """
    if model is None:
        model = LocalLiquidModel(
            LFM_LARGE,
            max_new_tokens=2048,
            temperature=0.1,  # Low temp for deterministic chunking
        )

    agent = Agent(
        model=model.get_pydantic_model(),
        output_type=ChunkingOutput,
        instructions=CHUNKING_INSTRUCTIONS,
    )

    return agent


# Pre-configured agent (lazy loaded)
_chunking_agent = None


def get_chunking_agent() -> Agent:
    """Get or create the global chunking agent."""
    global _chunking_agent
    if _chunking_agent is None:
        _chunking_agent = create_chunking_agent()
    return _chunking_agent


async def chunk_document(
    text: str,
    agent: Agent = None,
    max_chunks: int = 100,
) -> ChunkingOutput:
    """
    Chunk a document using the intelligent chunking agent.
    
    Args:
        text: Raw document text
        agent: Optional agent instance
        max_chunks: Maximum number of chunks to produce
        
    Returns:
        ChunkingOutput with list of chunks
    """
    if agent is None:
        agent = get_chunking_agent()

    # For very long documents, we may need to chunk in batches
    # This is a simple implementation; production would need sliding window

    prompt = f"""Please analyze and chunk the following document:

---
{text[:32000]}
---

Split this into semantically coherent chunks. Return as JSON with a 'chunks' array."""

    result = await agent.run(
        prompt,
        model_settings=ModelSettings(
            extra_body={"max_new_tokens": 4096}
        )
    )

    return result.output


async def chunk_document(text: str, agent: Agent = None) -> ChunkingOutput:
    """Async version of chunk_document."""
    if agent is None:
        agent = get_chunking_agent()

    prompt = f"""Please analyze and chunk the following document:

---
{text[:32000]}
---

Split this into semantically coherent chunks. Return as JSON with a 'chunks' array."""

    result = await agent.run(
        prompt,
        model_settings=ModelSettings(
            extra_body={"max_new_tokens": 4096}
        )
    )

    return result.data


def chunk_document_sync(text: str, agent: Agent = None) -> ChunkingOutput:
    """
    Synchronous wrapper for chunk_document.

    NOTE: This will only work if there's no running event loop.
    For use within async contexts (like pydantic-graph), use chunk_document() instead.
    """
    import asyncio

    try:
        # Try to get the running loop
        loop = asyncio.get_running_loop()
        # If we get here, there's a running loop - we can't use asyncio.run()
        raise RuntimeError(
            "chunk_document_sync() cannot be called from an async context. "
            "Use 'await chunk_document()' instead."
        )
    except RuntimeError:
        # No running loop, safe to use asyncio.run()
        return asyncio.run(chunk_document(text, agent))
