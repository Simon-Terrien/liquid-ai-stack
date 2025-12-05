# etl_pipeline/agents/validate_agent.py
"""
Validation agent using LiquidAI 700M.

Filters QA pairs to ensure quality:
- Removes hallucinated answers
- Ensures grounding in source chunks
- Checks question specificity
- Scores quality
"""

from liquid_shared import LFM_SMALL, Chunk, LocalLiquidModel, QAPair
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings


class ValidationResult(BaseModel):
    """Validation result for a single QA pair."""
    qa_index: int = Field(description="Index of the QA pair")
    is_valid: bool = Field(description="Whether the QA pair passes validation")
    quality_score: float = Field(ge=0, le=1, description="Quality score 0-1")
    issues: list[str] = Field(default_factory=list, description="List of issues found")


class ValidationOutput(BaseModel):
    """Output from validation."""
    results: list[ValidationResult] = Field(description="Validation results")


VALIDATION_INSTRUCTIONS = """You are an expert QA quality assessor. Your task is to validate question-answer pairs for training data quality.

For each QA pair, check:

1. **Grounding** (Critical):
   - Is the answer derivable from the source chunk?
   - Are there any facts in the answer not present in the chunk?
   - Flag: "hallucination" or "ungrounded_claim"

2. **Question Quality**:
   - Is the question specific and clear?
   - Would a user realistically ask this?
   - Flag: "vague_question", "unrealistic_question"

3. **Answer Quality**:
   - Is the answer complete and accurate?
   - Is it appropriately detailed (not too brief or verbose)?
   - Flag: "incomplete_answer", "too_brief", "too_verbose"

4. **Coherence**:
   - Does the answer actually address the question?
   - Is the language clear and professional?
   - Flag: "mismatched_qa", "unclear_language"

Quality Score Guidelines:
- 0.9-1.0: Excellent, no issues
- 0.7-0.9: Good, minor issues
- 0.5-0.7: Acceptable, some concerns
- 0.3-0.5: Poor, significant issues
- 0.0-0.3: Reject, major problems

Mark is_valid=true only for scores >= 0.6

Return validation results for all QA pairs.

Example output:
{
  "results": [
    {
      "qa_index": 0,
      "is_valid": true,
      "quality_score": 0.85,
      "issues": []
    },
    {
      "qa_index": 1,
      "is_valid": false,
      "quality_score": 0.4,
      "issues": ["hallucination: answer mentions deadline not in chunk", "incomplete_answer"]
    }
  ]
}
"""


def create_validate_agent(model: LocalLiquidModel = None) -> Agent:
    """
    Create the validation agent.
    
    Uses LFM2-700M for fast validation checks.
    """
    if model is None:
        model = LocalLiquidModel(
            LFM_SMALL,
            max_new_tokens=1024,
            temperature=0.1,  # Low temp for consistent validation
        )

    agent = Agent(
        model=model.get_pydantic_model(),
        output_type=ValidationOutput,
        instructions=VALIDATION_INSTRUCTIONS,
    )

    return agent


_validate_agent = None


def get_validate_agent() -> Agent:
    """Get or create the global validation agent."""
    global _validate_agent
    if _validate_agent is None:
        _validate_agent = create_validate_agent()
    return _validate_agent


async def validate_qa_pairs(
    qa_pairs: list[QAPair],
    chunks: list[Chunk],
    agent: Agent = None,
    min_quality: float = 0.6,
) -> list[QAPair]:
    """
    Validate and filter QA pairs.
    
    Args:
        qa_pairs: List of QA pairs to validate
        chunks: Source chunks for grounding check
        agent: Optional agent instance
        min_quality: Minimum quality score to keep (default 0.6)
        
    Returns:
        Filtered list of valid QA pairs with quality scores
    """
    if agent is None:
        agent = get_validate_agent()

    if not qa_pairs:
        return []

    # Build chunk lookup
    chunk_by_id = {c.id: c for c in chunks}

    # Format for validation
    formatted_items = []
    for i, qa in enumerate(qa_pairs):
        chunk = chunk_by_id.get(qa.chunk_id)
        chunk_text = chunk.text[:1000] if chunk else "[CHUNK NOT FOUND]"

        formatted_items.append(f"""[QA {i}]
SOURCE CHUNK:
{chunk_text}

QUESTION: {qa.question}

ANSWER: {qa.answer}
[/QA {i}]""")

    items_text = "\n\n".join(formatted_items)

    prompt = f"""Validate these {len(qa_pairs)} QA pairs against their source chunks.

Check for grounding, question quality, answer quality, and coherence.

{items_text}

Return validation results for all {len(qa_pairs)} QA pairs."""

    result = await agent.run(
        prompt,
        model_settings=ModelSettings(
            extra_body={"max_new_tokens": 2048}
        )
    )

    # Filter and update QA pairs
    validated_pairs = []
    for i, qa in enumerate(qa_pairs):
        # Find validation result
        val_result = None
        for r in result.output.results:
            if r.qa_index == i:
                val_result = r
                break

        if val_result and val_result.is_valid and val_result.quality_score >= min_quality:
            qa.quality_score = val_result.quality_score
            validated_pairs.append(qa)

    return validated_pairs


def validate_qa_pairs_sync(
    qa_pairs: list[QAPair],
    chunks: list[Chunk],
    agent: Agent = None,
    min_quality: float = 0.6,
) -> list[QAPair]:
    """Synchronous version of validate_qa_pairs."""
    if agent is None:
        agent = get_validate_agent()

    if not qa_pairs:
        return []

    chunk_by_id = {c.id: c for c in chunks}

    formatted_items = []
    for i, qa in enumerate(qa_pairs):
        chunk = chunk_by_id.get(qa.chunk_id)
        chunk_text = chunk.text[:1000] if chunk else "[CHUNK NOT FOUND]"

        formatted_items.append(f"""[QA {i}]
SOURCE CHUNK:
{chunk_text}

QUESTION: {qa.question}

ANSWER: {qa.answer}
[/QA {i}]""")

    items_text = "\n\n".join(formatted_items)

    prompt = f"""Validate these {len(qa_pairs)} QA pairs against their source chunks.

Check for grounding, question quality, answer quality, and coherence.

{items_text}

Return validation results for all {len(qa_pairs)} QA pairs."""

    result = agent.run_sync(
        prompt,
        model_settings=ModelSettings(
            extra_body={"max_new_tokens": 2048}
        )
    )

    validated_pairs = []
    for i, qa in enumerate(qa_pairs):
        val_result = None
        for r in result.output.results:
            if r.qa_index == i:
                val_result = r
                break

        if val_result and val_result.is_valid and val_result.quality_score >= min_quality:
            qa.quality_score = val_result.quality_score
            validated_pairs.append(qa)

    return validated_pairs
