# etl_pipeline/agents/qa_agent.py
"""
QA pair generation agent using LiquidAI 2.6B.

Generates question-answer pairs from document chunks for fine-tuning.
Each QA pair is grounded strictly in the source chunk content.
"""

from liquid_shared import LFM_MEDIUM, Chunk, LocalLiquidModel, QAPair
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings


class QAItem(BaseModel):
    """A single QA pair."""
    chunk_index: int = Field(description="Index of source chunk")
    question: str = Field(description="Natural user question")
    answer: str = Field(description="Answer grounded in chunk content")
    question_type: str = Field(
        description="Type: factual, definitional, procedural, comparative, analytical"
    )


class QAOutput(BaseModel):
    """Output from QA generation."""
    qa_pairs: list[QAItem] = Field(description="Generated QA pairs")


QA_INSTRUCTIONS = """You are an expert at generating high-quality question-answer pairs for training AI assistants.

For each chunk provided, generate 3-5 diverse QA pairs that:

1. **Are Grounded**: The answer MUST be derivable from the chunk text. No external knowledge.

2. **Are Realistic**: Questions should sound like real user queries, not exam questions.
   - Good: "What are the requirements for obtaining consent under GDPR?"
   - Bad: "List the requirements mentioned in paragraph 3."

3. **Cover Different Types**:
   - **factual**: "What is X?" "How many Y?"
   - **definitional**: "What does X mean?" "Define Y."
   - **procedural**: "How do I X?" "What are the steps for Y?"
   - **comparative**: "What's the difference between X and Y?"
   - **analytical**: "Why is X important?" "What are the implications of Y?"

4. **Have Complete Answers**: Answers should be:
   - 1-3 sentences for simple facts
   - A short paragraph for explanations
   - Include relevant context
   - Quote key terms when appropriate

5. **Avoid**:
   - Yes/no questions
   - Questions answerable without the chunk
   - Overly specific questions about document structure
   - Questions requiring information not in the chunk

Return valid JSON with QA pairs for all chunks.

Example output:
{
  "qa_pairs": [
    {
      "chunk_index": 0,
      "question": "What are the six lawful bases for processing personal data under GDPR?",
      "answer": "Under GDPR Article 6, the six lawful bases for processing personal data are: (1) consent from the data subject, (2) necessity for contract performance, (3) compliance with a legal obligation, (4) protection of vital interests, (5) performance of a public task, and (6) legitimate interests pursued by the controller.",
      "question_type": "factual"
    },
    {
      "chunk_index": 0,
      "question": "When can an organization rely on legitimate interests as a legal basis for data processing?",
      "answer": "An organization can rely on legitimate interests when processing is necessary for purposes pursued by the controller or a third party, except where such interests are overridden by the fundamental rights and freedoms of the data subject. This requires conducting a legitimate interests assessment (LIA) to balance the organization's interests against the individual's rights.",
      "question_type": "procedural"
    }
  ]
}
"""


def create_qa_agent(model: LocalLiquidModel = None) -> Agent:
    """
    Create the QA generation agent.
    
    Uses LFM2-2.6B for high-quality question generation.
    """
    if model is None:
        model = LocalLiquidModel(
            LFM_MEDIUM,  # Using 1.2B - optimal for data extraction per LFM2 docs
            max_new_tokens=2048,
            temperature=0.3,  # Slightly higher for diversity
        )

    agent = Agent(
        model=model.get_pydantic_model(),
        output_type=QAOutput,
        instructions=QA_INSTRUCTIONS,
    )

    return agent


_qa_agent = None


def get_qa_agent() -> Agent:
    """Get or create the global QA agent."""
    global _qa_agent
    if _qa_agent is None:
        _qa_agent = create_qa_agent()
    return _qa_agent


async def generate_qa_pairs(
    chunks: list[Chunk],
    agent: Agent = None,
    pairs_per_chunk: int = 4,
) -> list[QAPair]:
    """
    Generate QA pairs from chunks.
    
    Args:
        chunks: List of Chunk objects
        agent: Optional agent instance
        pairs_per_chunk: Target number of QA pairs per chunk
        
    Returns:
        List of QAPair objects
    """
    if agent is None:
        agent = get_qa_agent()

    # Format chunks
    formatted_chunks = []
    for chunk in chunks:
        header = f"[CHUNK {chunk.chunk_index}]"
        if chunk.section_title:
            header += f" - {chunk.section_title}"
        if chunk.tags:
            header += f" (Tags: {', '.join(chunk.tags[:5])})"
        formatted_chunks.append(f"{header}\n{chunk.text}\n[/CHUNK {chunk.chunk_index}]")

    chunks_text = "\n\n".join(formatted_chunks)

    prompt = f"""Generate {pairs_per_chunk} diverse QA pairs for each of these {len(chunks)} chunks:

{chunks_text}

Remember:
- Questions should be realistic user queries
- Answers must be grounded ONLY in the chunk content
- Cover different question types (factual, definitional, procedural, etc.)
- Make answers complete and informative"""

    result = await agent.run(
        prompt,
        model_settings=ModelSettings(
            extra_body={"max_new_tokens": 4096}
        )
    )

    # Convert to QAPair objects
    qa_pairs = []
    for qa in result.output.qa_pairs:
        # Find the corresponding chunk
        chunk = next((c for c in chunks if c.chunk_index == qa.chunk_index), None)
        if chunk:
            qa_pairs.append(QAPair(
                chunk_id=chunk.id,
                question=qa.question,
                answer=qa.answer,
            ))

    return qa_pairs


def generate_qa_pairs_sync(
    chunks: list[Chunk],
    agent: Agent = None,
    pairs_per_chunk: int = 4,
) -> list[QAPair]:
    """Synchronous version of generate_qa_pairs."""
    if agent is None:
        agent = get_qa_agent()

    formatted_chunks = []
    for chunk in chunks:
        header = f"[CHUNK {chunk.chunk_index}]"
        if chunk.section_title:
            header += f" - {chunk.section_title}"
        if chunk.tags:
            header += f" (Tags: {', '.join(chunk.tags[:5])})"
        formatted_chunks.append(f"{header}\n{chunk.text}\n[/CHUNK {chunk.chunk_index}]")

    chunks_text = "\n\n".join(formatted_chunks)

    prompt = f"""Generate {pairs_per_chunk} diverse QA pairs for each of these {len(chunks)} chunks:

{chunks_text}

Remember:
- Questions should be realistic user queries
- Answers must be grounded ONLY in the chunk content
- Cover different question types (factual, definitional, procedural, etc.)
- Make answers complete and informative"""

    result = agent.run_sync(
        prompt,
        model_settings=ModelSettings(
            extra_body={"max_new_tokens": 4096}
        )
    )

    qa_pairs = []
    for qa in result.output.qa_pairs:
        chunk = next((c for c in chunks if c.chunk_index == qa.chunk_index), None)
        if chunk:
            qa_pairs.append(QAPair(
                chunk_id=chunk.id,
                question=qa.question,
                answer=qa.answer,
            ))

    return qa_pairs
