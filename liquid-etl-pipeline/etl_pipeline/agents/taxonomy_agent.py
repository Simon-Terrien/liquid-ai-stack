# etl_pipeline/agents/taxonomy_agent.py
"""
Taxonomy extraction agent using LiquidAI 1.2B.

Extracts hierarchical taxonomy structures from document chunks.
Inspired by previous ETL project's rich taxonomy extraction.
"""

from liquid_shared import LFM_MEDIUM, Chunk, LocalLiquidModel
from liquid_shared.schemas import TaxonomyNode
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings


class ChunkTaxonomy(BaseModel):
    """Taxonomy output for a single chunk."""
    chunk_index: int = Field(description="Index of the chunk")
    taxonomy: TaxonomyNode = Field(
        description="Hierarchical taxonomy structure for this chunk"
    )


class TaxonomyOutput(BaseModel):
    """Output from taxonomy extraction agent."""
    taxonomies: list[ChunkTaxonomy] = Field(
        description="Taxonomies for all chunks"
    )


TAXONOMY_INSTRUCTIONS = """You are an expert at extracting hierarchical topic taxonomies from text.

For each chunk provided, identify the main topic and its subtopics in a hierarchical tree structure.

**Taxonomy Guidelines**:

1. **Identify Main Topic**: What is the primary subject of this chunk?
   - This becomes the root node of the taxonomy

2. **Find Subtopics**: What are 2-4 key subtopics or aspects discussed?
   - These become child nodes

3. **Go Deeper**: For important subtopics, identify their sub-subtopics
   - Create a tree 2-3 levels deep
   - Don't go too deep - keep it practical

4. **Assign Importance**:
   - "High": Critical concepts, core principles, key definitions
   - "Medium": Supporting concepts, important details
   - "Low": Minor details, examples, context

5. **Categorize Each Node**:
   - Technical: Technical implementations, systems, methods
   - Conceptual: Theories, principles, frameworks
   - Regulatory: Laws, policies, compliance requirements
   - Operational: Processes, procedures, workflows
   - Threats: Risks, attacks, vulnerabilities
   - Defense: Protections, controls, mitigations

**Example Structure**:

```
AI Security (High, Technical)
├── Adversarial Attacks (High, Threats)
│   ├── Evasion Attacks (Medium, Threats)
│   └── Poisoning Attacks (Medium, Threats)
├── Model Robustness (High, Defense)
│   ├── Adversarial Training (Medium, Defense)
│   └── Input Validation (Medium, Defense)
└── Monitoring (Medium, Operational)
```

**Output Format**:

Return a TaxonomyNode with:
- `name`: Topic name (2-5 words)
- `description`: Brief description of what this topic covers (optional, 1 sentence)
- `importance`: "High", "Medium", or "Low"
- `category`: One of: Technical, Conceptual, Regulatory, Operational, Threats, Defense
- `children`: List of child TaxonomyNode objects (can be empty)

**Example Output**:

```json
{
  "taxonomies": [
    {
      "chunk_index": 0,
      "taxonomy": {
        "name": "AI Security",
        "description": "Security considerations for AI/ML systems",
        "importance": "High",
        "category": "Technical",
        "children": [
          {
            "name": "Adversarial Attacks",
            "description": "Attacks that manipulate model behavior",
            "importance": "High",
            "category": "Threats",
            "children": [
              {
                "name": "Evasion Attacks",
                "description": "Attacks during inference to cause misclassification",
                "importance": "Medium",
                "category": "Threats",
                "children": []
              }
            ]
          },
          {
            "name": "Model Robustness",
            "description": "Techniques to make models resilient to attacks",
            "importance": "High",
            "category": "Defense",
            "children": []
          }
        ]
      }
    }
  ]
}
```

Return valid JSON with taxonomies for each chunk.
"""


def create_taxonomy_agent(model: LocalLiquidModel = None) -> Agent:
    """
    Create the taxonomy extraction agent.

    Uses LFM2-1.2B for quality taxonomy extraction.

    Args:
        model: Optional pre-loaded model. If None, loads LFM2-1.2B.

    Returns:
        Configured Pydantic AI Agent
    """
    if model is None:
        model = LocalLiquidModel(
            LFM_MEDIUM,  # Using 1.2B - good balance for taxonomy extraction
            max_new_tokens=1024,
            temperature=0.2,  # Some creativity for finding relationships
        )

    agent = Agent(
        model=model.get_pydantic_model(),
        output_type=TaxonomyOutput,
        instructions=TAXONOMY_INSTRUCTIONS,
    )

    return agent


def get_taxonomy_agent() -> Agent:
    """Get or create the global taxonomy agent instance."""
    return create_taxonomy_agent()


async def extract_taxonomies(chunks: list[Chunk], agent: Agent = None) -> TaxonomyOutput:
    """
    Extract hierarchical taxonomies from chunks.

    Args:
        chunks: List of chunks to extract taxonomies from
        agent: Optional pre-configured agent

    Returns:
        TaxonomyOutput with hierarchical structures for each chunk
    """
    if agent is None:
        agent = get_taxonomy_agent()

    # Prepare chunks for taxonomy extraction
    chunk_texts = []
    for i, chunk in enumerate(chunks):
        # Include metadata to help with taxonomy extraction
        chunk_text = f"""
**Chunk {i}** (importance: {chunk.importance}/10)
Section: {chunk.section_title or 'Unknown'}
Tags: {', '.join(chunk.tags) if chunk.tags else 'None'}
Keywords: {', '.join(chunk.keywords) if chunk.keywords else 'None'}
Categories: {', '.join(chunk.categories) if chunk.categories else 'None'}

{chunk.text[:2000]}...
"""
        chunk_texts.append(chunk_text)

    prompt = f"""Extract hierarchical topic taxonomies for the following {len(chunks)} chunks:

{chr(10).join(chunk_texts)}

Return taxonomies for all {len(chunks)} chunks."""

    result = await agent.run(
        prompt,
        model_settings=ModelSettings(extra_body={"max_new_tokens": 2048})
    )

    return result.output


def extract_taxonomies_sync(chunks: list[Chunk], agent: Agent = None) -> TaxonomyOutput:
    """
    Synchronous wrapper for extract_taxonomies.

    NOTE: This will only work if there's no running event loop.
    For use within async contexts (like pydantic-graph), use extract_taxonomies() instead.
    """
    import asyncio
    try:
        loop = asyncio.get_running_loop()
        raise RuntimeError(
            "extract_taxonomies_sync() cannot be called from an async context. "
            "Use 'await extract_taxonomies()' instead."
        )
    except RuntimeError:
        return asyncio.run(extract_taxonomies(chunks, agent))


if __name__ == "__main__":
    # Quick test
    import asyncio
    from liquid_shared import Chunk

    test_chunk = Chunk(
        id="test-1",
        text="""Machine learning models are vulnerable to adversarial attacks where small
        perturbations to input data can cause misclassification. These attacks pose significant
        security risks in production AI systems. Organizations must implement robust defenses
        including adversarial training, input validation, and model monitoring to protect against
        these threats. The GDPR requires that AI systems processing personal data maintain
        appropriate security measures.""",
        section_title="AI Security Challenges",
        source_path="test.pdf",
        chunk_index=0,
        importance=8,
        tags=["ai", "security", "adversarial-attacks"],
        keywords=["adversarial attacks", "model robustness", "GDPR compliance"],
        entities=["Machine Learning", "GDPR"],
        categories=["AI/ML Security", "Data Protection"],
    )

    async def test():
        print("Testing taxonomy extraction...")
        print("=" * 70)
        result = await extract_taxonomies([test_chunk])

        if result.taxonomies:
            taxonomy = result.taxonomies[0].taxonomy
            print(f"\n✅ Taxonomy extracted successfully!\n")
            print(f"Root Topic: {taxonomy.name}")
            print(f"Importance: {taxonomy.importance}")
            print(f"Category: {taxonomy.category}")
            print(f"Children: {len(taxonomy.children)}")

            if taxonomy.children:
                print(f"\nSubtopics:")
                for child in taxonomy.children:
                    print(f"  - {child.name} ({child.importance}, {child.category})")
                    if child.children:
                        for subchild in child.children:
                            print(f"    - {subchild.name} ({subchild.importance}, {subchild.category})")
        else:
            print("❌ No taxonomies returned")

    asyncio.run(test())
