# etl_pipeline/agents/metadata_agent.py
"""
Metadata extraction agent using LiquidAI 2.6B.

Extracts structured metadata from document chunks:
- Section titles
- Topic tags
- Named entities
- Importance scores
"""

from liquid_shared import LFM_MEDIUM, Chunk, LocalLiquidModel
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings


class ChunkMetadata(BaseModel):
    """Extracted metadata for a single chunk."""
    chunk_index: int = Field(description="Index of the chunk (0-based)")
    section_title: str = Field(description="Concise title for this section")
    tags: list[str] = Field(description="Topic tags (3-7 tags)")
    entities: list[str] = Field(description="Named entities (orgs, laws, systems, people)")
    importance: int = Field(ge=0, le=10, description="Importance score 0-10")
    summary_hint: str = Field(description="Brief hint for what to emphasize in summary")


class MetadataOutput(BaseModel):
    """Output from metadata extraction."""
    metadata: list[ChunkMetadata] = Field(description="Metadata for each chunk")


METADATA_INSTRUCTIONS = """You are an expert at analyzing text and extracting structured metadata.

For each chunk provided, extract the following:

1. **section_title**: A concise, descriptive title (3-8 words) that captures the main topic.
   - Use title case
   - Be specific, not generic
   - Example: "GDPR Data Subject Rights Overview"

2. **tags**: 3-7 relevant topic tags
   - Use lowercase with hyphens
   - Include domain-specific terms
   - Examples: ["data-protection", "gdpr", "privacy-rights", "eu-law"]

3. **entities**: Named entities found in the text
   - Organizations (companies, agencies, bodies)
   - Laws and regulations (GDPR, CCPA, ISO 27001)
   - Systems and technologies
   - Key people mentioned
   - Standards and frameworks

4. **importance**: Score from 0-10
   - 0-2: Administrative, boilerplate
   - 3-4: Supporting context
   - 5-6: Standard content
   - 7-8: Key concepts, definitions
   - 9-10: Critical requirements, core principles

5. **summary_hint**: What should be emphasized when summarizing this chunk
   - Note key facts, requirements, or definitions
   - Identify the main takeaway

Return valid JSON with metadata for each chunk, preserving chunk order.

Example output:
{
  "metadata": [
    {
      "chunk_index": 0,
      "section_title": "Introduction to Data Protection",
      "tags": ["data-protection", "privacy", "compliance"],
      "entities": ["GDPR", "European Commission", "Data Protection Authority"],
      "importance": 7,
      "summary_hint": "Defines core data protection principles and scope"
    }
  ]
}
"""


def create_metadata_agent(model: LocalLiquidModel = None) -> Agent:
    """
    Create the metadata extraction agent.
    
    Args:
        model: Optional pre-loaded model. If None, loads LFM2-2.6B.
        
    Returns:
        Configured Pydantic AI Agent
    """
    if model is None:
        model = LocalLiquidModel(
            LFM_MEDIUM,  # Using 1.2B - optimal for data extraction per LFM2 docs
            max_new_tokens=2048,
            temperature=0.1,
        )

    agent = Agent(
        model=model.get_pydantic_model(),
        output_type=MetadataOutput,
        instructions=METADATA_INSTRUCTIONS,
    )

    return agent


_metadata_agent = None


def get_metadata_agent() -> Agent:
    """Get or create the global metadata agent."""
    global _metadata_agent
    if _metadata_agent is None:
        _metadata_agent = create_metadata_agent()
    return _metadata_agent


async def extract_metadata(
    chunks: list[str],
    source_path: str,
    agent: Agent = None,
) -> list[Chunk]:
    """
    Extract metadata from chunks.
    
    Args:
        chunks: List of chunk texts
        source_path: Path to source document
        agent: Optional agent instance
        
    Returns:
        List of Chunk objects with metadata
    """
    import uuid

    if agent is None:
        agent = get_metadata_agent()

    # Format chunks for the prompt
    formatted_chunks = []
    for i, chunk in enumerate(chunks):
        formatted_chunks.append(f"[CHUNK {i}]\n{chunk[:2000]}\n[/CHUNK {i}]")

    chunks_text = "\n\n".join(formatted_chunks)

    prompt = f"""Analyze the following {len(chunks)} chunks and extract metadata for each:

{chunks_text}

Return metadata for all {len(chunks)} chunks in order."""

    result = await agent.run(
        prompt,
        model_settings=ModelSettings(
            extra_body={"max_new_tokens": 4096}
        )
    )

    # Convert to Chunk objects
    output_chunks = []
    for i, chunk_text in enumerate(chunks):
        # Find corresponding metadata
        meta = None
        for m in result.output.metadata:
            if m.chunk_index == i:
                meta = m
                break

        output_chunks.append(Chunk(
            id=str(uuid.uuid4()),
            text=chunk_text,
            section_title=meta.section_title if meta else None,
            source_path=source_path,
            chunk_index=i,
            importance=meta.importance if meta else 5,
            tags=meta.tags if meta else [],
            entities=meta.entities if meta else [],
        ))

    return output_chunks


def extract_metadata_sync(
    chunks: list[str],
    source_path: str,
    agent: Agent = None,
) -> list[Chunk]:
    """Synchronous version of extract_metadata."""
    import uuid

    if agent is None:
        agent = get_metadata_agent()

    formatted_chunks = []
    for i, chunk in enumerate(chunks):
        formatted_chunks.append(f"[CHUNK {i}]\n{chunk[:2000]}\n[/CHUNK {i}]")

    chunks_text = "\n\n".join(formatted_chunks)

    prompt = f"""Analyze the following {len(chunks)} chunks and extract metadata for each:

{chunks_text}

Return metadata for all {len(chunks)} chunks in order."""

    result = agent.run_sync(
        prompt,
        model_settings=ModelSettings(
            extra_body={"max_new_tokens": 4096}
        )
    )

    output_chunks = []
    for i, chunk_text in enumerate(chunks):
        meta = None
        for m in result.output.metadata:
            if m.chunk_index == i:
                meta = m
                break

        output_chunks.append(Chunk(
            id=str(uuid.uuid4()),
            text=chunk_text,
            section_title=meta.section_title if meta else None,
            source_path=source_path,
            chunk_index=i,
            importance=meta.importance if meta else 5,
            tags=meta.tags if meta else [],
            entities=meta.entities if meta else [],
        ))

    return output_chunks
