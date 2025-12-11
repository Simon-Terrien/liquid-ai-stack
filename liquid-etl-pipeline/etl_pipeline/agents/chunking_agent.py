# etl_pipeline/agents/chunking_agent.py
"""
Intelligent chunking agent using LiquidAI 2.6B.

Splits documents into semantically coherent chunks based on:
- Topic transitions
- Section boundaries
- Paragraph structure
- Logical coherence
"""

from liquid_shared import LFM_MEDIUM, LocalLiquidModel
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings


class ChunkBoundary(BaseModel):
    """Represents a chunk boundary in the document."""
    start_char: int = Field(description="Start character position")
    end_char: int = Field(description="End character position")
    reason: str = Field(description="Why this boundary was chosen")


class ChunkingAgentOutput(BaseModel):
    """Minimal output returned directly by the LLM."""
    boundaries: list[ChunkBoundary] = Field(
        default_factory=list,
        description="Boundary information for each chunk"
    )


class ChunkingOutput(BaseModel):
    """Output after post-processing the LLM response."""
    chunks: list[str] = Field(description="List of chunk texts")
    boundaries: list[ChunkBoundary] = Field(
        default_factory=list,
        description="Boundary information for each chunk"
    )


CHUNKING_INSTRUCTIONS = """You are an expert document analyzer specializing in semantic text segmentation.

Mark semantic chunk boundaries only. Do NOT return chunk text. Use character offsets on the raw text.

Rules for boundaries:
1. Respect natural transitions: headings, topic shifts, paragraph breaks that signal new ideas, list completions, definitions.
2. Preserve context: never split mid-sentence, inside a list item, between a heading and its content, or inside a code block/table row.
3. Size: aim for 500-1500 tokens per chunk; smaller is acceptable for short sections.
4. Keep boundaries ordered from start to end and non-empty (end_char > start_char).

Return ONLY valid JSON with this exact schema:
{
  "boundaries": [
    {"start_char": 0, "end_char": 500, "reason": "Introduction"}
  ]
}

No prose, no markdown, no chunk text.
"""


def create_chunking_agent(model: LocalLiquidModel = None) -> Agent:
    """
    Create the chunking agent.

    Args:
        model: Optional pre-loaded model. If None, loads LFM2-1.2B.

    Returns:
        Configured Pydantic AI Agent
    """
    if model is None:
        model = LocalLiquidModel(
            LFM_MEDIUM,  # Using 1.2B - optimal for data extraction per LFM2 docs
            max_new_tokens=2048,
            temperature=0.0,  # Deterministic chunking
        )

    agent = Agent(
        model=model.get_pydantic_model(),
        output_type=ChunkingAgentOutput,
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


def _normalize_boundaries(
    text: str,
    boundaries: list[ChunkBoundary],
    max_chunks: int,
) -> list[ChunkBoundary]:
    """Clamp, deduplicate, and truncate boundaries to fit the source text."""
    text_len = len(text)
    normalized: list[ChunkBoundary] = []

    for boundary in boundaries:
        start = max(0, min(text_len, boundary.start_char))
        end = max(0, min(text_len, boundary.end_char))
        if end <= start:
            continue
        normalized.append(
            ChunkBoundary(
                start_char=start,
                end_char=end,
                reason=boundary.reason,
            )
        )

    normalized.sort(key=lambda b: b.start_char)

    deduped: list[ChunkBoundary] = []
    last_end = -1
    for boundary in normalized:
        if boundary.start_char < last_end:
            # Skip overlaps to keep ordering clean
            continue
        deduped.append(boundary)
        last_end = boundary.end_char

    return deduped[:max_chunks]


def _boundaries_to_chunks(text: str, boundaries: list[ChunkBoundary]) -> list[str]:
    """Slice chunk texts from the original document using boundaries."""
    return [text[b.start_char:b.end_char] for b in boundaries]


async def chunk_document(
    text: str,
    agent: Agent = None,
    max_chunks: int = 100,
    max_prompt_chars: int = 32000,
) -> ChunkingOutput:
    """Chunk a document using the intelligent chunking agent.

    Falls back to returning an empty list if the agent returns no usable boundaries.
    """
    if agent is None:
        agent = get_chunking_agent()

    prompt = f"""Identify semantic chunk boundaries for the document below.
Return ONLY JSON with the 'boundaries' field. Do not include chunk text.

---
{text[:max_prompt_chars]}
---
"""

    result = await agent.run(
        prompt,
        model_settings=ModelSettings(
            extra_body={"max_new_tokens": 4096}
        ),
    )

    boundaries = _normalize_boundaries(text, result.output.boundaries, max_chunks)
    chunks = _boundaries_to_chunks(text, boundaries)

    return ChunkingOutput(chunks=chunks, boundaries=boundaries)


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
