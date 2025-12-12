# etl_pipeline/agents/classification_agent.py
"""
Classification agent using LiquidAI 700M for fast categorization.

Classifies document chunks into predefined categories based on content.
Inspired by previous ETL project's multi-label classification approach.
"""

from liquid_shared import LFM_SMALL, Chunk, LocalLiquidModel
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings


class ChunkClassification(BaseModel):
    """Classification output for a single chunk."""
    chunk_index: int = Field(description="Index of the chunk")
    categories: list[str] = Field(
        description="List of applicable categories for this chunk"
    )
    primary_category: str = Field(
        description="The most relevant category"
    )
    confidence: float = Field(
        ge=0, le=1,
        description="Confidence score for the classification"
    )


class ClassificationOutput(BaseModel):
    """Output from classification agent."""
    classifications: list[ChunkClassification] = Field(
        description="Classifications for all chunks"
    )


# Define cybersecurity/AI domain categories
# Based on the documents in data/raw/ (ENISA reports, AI cybersecurity)
CATEGORY_TAXONOMY = """
**Available Categories** (select 1-3 most relevant):

1. **AI/ML Security**
   - Machine learning vulnerabilities
   - AI model security
   - Adversarial attacks
   - Model robustness

2. **Cyber Insurance**
   - Risk assessment
   - Insurance policies
   - Coverage and claims
   - Premium calculation

3. **Data Protection**
   - GDPR compliance
   - Privacy regulations
   - Data governance
   - Personal data handling

4. **Threat Intelligence**
   - Cyber threats
   - Attack vectors
   - Threat actors
   - Incident response

5. **Risk Management**
   - Risk assessment frameworks
   - Cybersecurity controls
   - Compliance requirements
   - Security standards

6. **Technical Controls**
   - Security implementations
   - Authentication/authorization
   - Encryption
   - Network security

7. **Governance & Policy**
   - Security policies
   - Regulatory frameworks
   - Best practices
   - Guidelines and standards

8. **Research & Innovation**
   - Emerging technologies
   - Research findings
   - Future directions
   - Novel approaches
"""

CLASSIFICATION_INSTRUCTIONS = f"""You are an expert at classifying cybersecurity and AI-related content.

For each chunk provided, identify the 1-3 most relevant categories from the taxonomy below.
Also identify ONE primary category (the most relevant).

{CATEGORY_TAXONOMY}

**Classification Guidelines**:
1. Read the chunk carefully to understand its main focus
2. Select categories that match the core content (not just mentions)
3. Prioritize specificity - choose the most specific applicable category
4. Multi-label: A chunk can belong to multiple categories
5. Primary category: Pick the single most dominant theme

**Confidence Scoring**:
- 0.9-1.0: Chunk is clearly and primarily about these categories
- 0.7-0.8: Chunk is substantially about these categories with some other content
- 0.5-0.6: Chunk touches on these categories among other topics
- Below 0.5: Uncertain categorization

Return valid JSON with classifications for each chunk.

Example output:
{{
  "classifications": [
    {{
      "chunk_index": 0,
      "categories": ["AI/ML Security", "Risk Management"],
      "primary_category": "AI/ML Security",
      "confidence": 0.85
    }}
  ]
}}
"""


def create_classification_agent(model: LocalLiquidModel = None) -> Agent:
    """
    Create the classification agent.

    Uses LFM2-700M for fast, efficient classification.

    Args:
        model: Optional pre-loaded model. If None, loads LFM2-700M.

    Returns:
        Configured Pydantic AI Agent
    """
    if model is None:
        model = LocalLiquidModel(
            LFM_SMALL,  # Using 700M - fast classification
            max_new_tokens=512,
            temperature=0.1,  # Low temp for consistent categorization
        )

    agent = Agent(
        model=model.get_pydantic_model(),
        output_type=ClassificationOutput,
        instructions=CLASSIFICATION_INSTRUCTIONS,
    )

    return agent


def get_classification_agent() -> Agent:
    """Get or create the global classification agent instance."""
    return create_classification_agent()


async def classify_chunks(chunks: list[Chunk], agent: Agent = None) -> ClassificationOutput:
    """
    Classify a list of chunks into categories.

    Args:
        chunks: List of chunks to classify
        agent: Optional pre-configured agent

    Returns:
        ClassificationOutput with categories for each chunk
    """
    if agent is None:
        agent = get_classification_agent()

    # Prepare chunks for classification
    chunk_texts = []
    for i, chunk in enumerate(chunks):
        chunk_text = f"""
**Chunk {i}** (importance: {chunk.importance}/10)
Section: {chunk.section_title or 'Unknown'}
Tags: {', '.join(chunk.tags) if chunk.tags else 'None'}

{chunk.text[:2000]}...
"""
        chunk_texts.append(chunk_text)

    prompt = f"""Classify the following {len(chunks)} chunks into appropriate categories:

{chr(10).join(chunk_texts)}

Return classifications for all {len(chunks)} chunks."""

    result = await agent.run(
        prompt,
        model_settings=ModelSettings(extra_body={"max_new_tokens": 512})
    )

    return result.output


def classify_chunks_sync(chunks: list[Chunk], agent: Agent = None) -> ClassificationOutput:
    """
    Synchronous wrapper for classify_chunks.

    NOTE: This will only work if there's no running event loop.
    For use within async contexts (like pydantic-graph), use classify_chunks() instead.
    """
    import asyncio
    try:
        loop = asyncio.get_running_loop()
        raise RuntimeError(
            "classify_chunks_sync() cannot be called from an async context. "
            "Use 'await classify_chunks()' instead."
        )
    except RuntimeError:
        return asyncio.run(classify_chunks(chunks, agent))


if __name__ == "__main__":
    # Quick test
    import asyncio
    from liquid_shared import Chunk

    test_chunk = Chunk(
        id="test-1",
        text="Machine learning models are vulnerable to adversarial attacks where small perturbations to input data can cause misclassification. These attacks pose significant security risks in production AI systems.",
        section_title="AI Security Challenges",
        source_path="test.pdf",
        chunk_index=0,
        importance=8,
        tags=["ai", "security", "adversarial-attacks"],
        entities=["Machine Learning"],
    )

    async def test():
        result = await classify_chunks([test_chunk])
        print(f"Categories: {result.classifications[0].categories}")
        print(f"Primary: {result.classifications[0].primary_category}")
        print(f"Confidence: {result.classifications[0].confidence}")

    asyncio.run(test())
