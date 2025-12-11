# etl_pipeline/agents/summary_agent.py
"""
Summary agent using LiquidAI 1.2B.

Generates two types of text for each chunk:
1. Human-readable summary (2-4 sentences)
2. Embedding-optimized text (factual, dense, retrieval-friendly)
"""

from liquid_shared import LFM_SMALL, Chunk, ChunkSummary, LocalLiquidModel
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings


class ChunkSummaryItem(BaseModel):
    """Summary output for a single chunk."""
    chunk_index: int = Field(description="Index of the source chunk")
    summary: str = Field(description="Human-readable 2-4 sentence summary")
    embedding_text: str = Field(description="Embedding-optimized text for retrieval")


class SummaryOutput(BaseModel):
    """Output from summary generation."""
    summaries: list[ChunkSummaryItem] = Field(description="Summaries for all chunks")


SUMMARY_INSTRUCTIONS = """You are an expert at creating concise, accurate summaries optimized for different purposes.

For each chunk, generate TWO versions:

1. **summary** (Human-Readable):
   - 2-4 complete sentences
   - Natural, flowing prose
   - Captures the main point and key details
   - Written for humans to read and understand quickly
   - Example: "This section defines the lawful bases for processing personal data under GDPR. The six legal grounds include consent, contract necessity, legal obligation, vital interests, public task, and legitimate interests. Each basis has specific requirements and limitations that organizations must consider."

2. **embedding_text** (Retrieval-Optimized):
   - Dense with key facts and terms
   - Includes all acronyms, technical terms, and entity names
   - Removes filler words and transitions
   - Structured for maximum semantic signal
   - Example: "GDPR lawful processing bases: consent, contract necessity, legal obligation, vital interests, public task, legitimate interests. Article 6 requirements. Data controller obligations. Legal grounds for personal data processing EU regulation."

Key differences:
- Summary: Readable, contextual, complete sentences
- Embedding: Keyword-rich, factual, retrieval-focused

Return valid JSON with both versions for each chunk.

Example output:
{
  "summaries": [
    {
      "chunk_index": 0,
      "summary": "This introduction explains the purpose and scope of the GDPR regulation...",
      "embedding_text": "GDPR General Data Protection Regulation EU 2016/679 scope territorial application..."
    }
  ]
}
"""


def create_summary_agent(model: LocalLiquidModel = None) -> Agent:
    """
    Create the summary agent.
    
    Uses LFM2-1.2B for balanced speed/quality in summarization.
    """
    if model is None:
        model = LocalLiquidModel(
            LFM_SMALL,  # Using 700M - fast and efficient for summarization
            max_new_tokens=512,
            temperature=0.2,
        )

    agent = Agent(
        model=model.get_pydantic_model(),
        output_type=SummaryOutput,
        instructions=SUMMARY_INSTRUCTIONS,
    )

    return agent


_summary_agent = None


def get_summary_agent() -> Agent:
    """Get or create the global summary agent."""
    global _summary_agent
    if _summary_agent is None:
        _summary_agent = create_summary_agent()
    return _summary_agent


async def generate_summaries(
    chunks: list[Chunk],
    agent: Agent = None,
) -> list[ChunkSummary]:
    """
    Generate summaries for chunks.
    
    Args:
        chunks: List of Chunk objects with text
        agent: Optional agent instance
        
    Returns:
        List of ChunkSummary objects
    """
    if agent is None:
        agent = get_summary_agent()

    # Format chunks for the prompt
    formatted_chunks = []
    for chunk in chunks:
        header = f"[CHUNK {chunk.chunk_index}]"
        if chunk.section_title:
            header += f" - {chunk.section_title}"
        formatted_chunks.append(f"{header}\n{chunk.text[:1500]}\n[/CHUNK {chunk.chunk_index}]")

    chunks_text = "\n\n".join(formatted_chunks)

    prompt = f"""Generate summaries for the following {len(chunks)} chunks:

{chunks_text}

Create both a human-readable summary and embedding-optimized text for each chunk."""

    result = await agent.run(
        prompt,
        model_settings=ModelSettings(
            extra_body={"max_new_tokens": 2048}
        )
    )

    # Map back to ChunkSummary objects
    summaries = []
    for chunk in chunks:
        # Find matching summary
        summary_item = None
        for s in result.output.summaries:
            if s.chunk_index == chunk.chunk_index:
                summary_item = s
                break

        if summary_item:
            summaries.append(ChunkSummary(
                chunk_id=chunk.id,
                summary=summary_item.summary,
                embedding_text=summary_item.embedding_text,
            ))
        else:
            # Fallback: use chunk text directly
            summaries.append(ChunkSummary(
                chunk_id=chunk.id,
                summary=chunk.text[:500],
                embedding_text=chunk.text[:300],
            ))

    return summaries


def generate_summaries_sync(
    chunks: list[Chunk],
    agent: Agent = None,
) -> list[ChunkSummary]:
    """Synchronous version of generate_summaries."""
    if agent is None:
        agent = get_summary_agent()

    formatted_chunks = []
    for chunk in chunks:
        header = f"[CHUNK {chunk.chunk_index}]"
        if chunk.section_title:
            header += f" - {chunk.section_title}"
        formatted_chunks.append(f"{header}\n{chunk.text[:1500]}\n[/CHUNK {chunk.chunk_index}]")

    chunks_text = "\n\n".join(formatted_chunks)

    prompt = f"""Generate summaries for the following {len(chunks)} chunks:

{chunks_text}

Create both a human-readable summary and embedding-optimized text for each chunk."""

    result = agent.run_sync(
        prompt,
        model_settings=ModelSettings(
            extra_body={"max_new_tokens": 2048}
        )
    )

    summaries = []
    for chunk in chunks:
        summary_item = None
        for s in result.output.summaries:
            if s.chunk_index == chunk.chunk_index:
                summary_item = s
                break

        if summary_item:
            summaries.append(ChunkSummary(
                chunk_id=chunk.id,
                summary=summary_item.summary,
                embedding_text=summary_item.embedding_text,
            ))
        else:
            summaries.append(ChunkSummary(
                chunk_id=chunk.id,
                summary=chunk.text[:500],
                embedding_text=chunk.text[:300],
            ))

    return summaries
