"""
Enhanced QA Generation Agent with Domain-Specific Questions.

Generates multiple types of questions:
1. NER Questions (Named entities: people, orgs, technologies, vulnerabilities)
2. Technical Questions (How it works, components, mechanisms)
3. Domain Questions (Risks, mitigations, best practices)
4. Conceptual Questions (Definitions, relationships, implications)

Target: 5-10 QA pairs per chunk (vs 3 per document in original)
"""
from typing import List

from liquid_shared import Chunk, QAPair, LFM_LARGE, LocalLiquidModel
from pydantic import BaseModel, Field
from pydantic_ai import Agent


class EnhancedQAOutput(BaseModel):
    """Enhanced QA pairs with question type classification."""

    ner_questions: List[QAPair] = Field(
        default_factory=list,
        description="Named Entity Recognition questions (What is X? Who is Y?)"
    )
    technical_questions: List[QAPair] = Field(
        default_factory=list,
        description="Technical/mechanism questions (How does X work?)"
    )
    domain_questions: List[QAPair] = Field(
        default_factory=list,
        description="Domain-specific questions (What are risks of X? How to mitigate Y?)"
    )
    conceptual_questions: List[QAPair] = Field(
        default_factory=list,
        description="Conceptual/definitional questions"
    )


ENHANCED_QA_INSTRUCTIONS = """You are an expert in cybersecurity and AI, creating high-quality question-answer pairs for fine-tuning a domain-specific LLM.

Generate 5-10 QA pairs per chunk covering these categories:

**1. NER Questions (Named Entity Recognition)**
Extract and question key entities:
- Technologies: "What is adversarial machine learning?"
- Organizations: "What does ENISA stand for?"
- Vulnerabilities: "What is a model poisoning attack?"
- Standards: "What is GDPR?"
- People/Roles: "What is the role of a Chief Information Security Officer?"

**2. Technical Questions**
Focus on mechanisms and how things work:
- "How does differential privacy protect data?"
- "What are the components of a threat modeling framework?"
- "How can adversarial examples evade ML classifiers?"
- "What techniques are used for secure model training?"

**3. Domain Questions**
Cybersecurity and AI-specific:
- "What are the main risks of deploying AI systems in production?"
- "How can organizations mitigate supply chain attacks on ML models?"
- "What are best practices for AI governance?"
- "What vulnerabilities are unique to federated learning?"

**4. Conceptual Questions**
Definitions and relationships:
- "What is the difference between adversarial robustness and model accuracy?"
- "How does AI system explainability relate to security?"
- "What is the relationship between privacy and model utility?"

**Guidelines:**
- Questions should be answerable from the chunk text
- Answers should be 1-3 sentences, factual, and precise
- Prefer technical terminology over generic language
- Include acronyms and their expansions
- Focus on cybersecurity/AI domain knowledge
- Vary question types across all 4 categories

**Quality criteria:**
- Specificity: Questions should be precise, not vague
- Answerability: Answer must be in the chunk
- Domain relevance: Focus on cybersecurity/AI topics
- Educational value: Would this help an LLM learn domain knowledge?

Return JSON with categorized questions.
"""


def create_enhanced_qa_agent(model: LocalLiquidModel = None) -> Agent:
    """
    Create enhanced QA generation agent.

    Uses LFM2-2.6B for highest quality question generation.
    """
    if model is None:
        model = LocalLiquidModel(
            LFM_LARGE,  # 2.6B for quality
            max_new_tokens=4096,  # More tokens for multiple QA pairs
            temperature=0.3,  # Slightly creative for question diversity
        )

    agent = Agent(
        model,
        result_type=EnhancedQAOutput,
        system_prompt=ENHANCED_QA_INSTRUCTIONS,
    )

    return agent


async def generate_enhanced_qa_pairs(chunks: List[Chunk]) -> List[QAPair]:
    """
    Generate enhanced QA pairs from chunks.

    Creates 5-10 QA pairs per chunk across 4 categories:
    - NER questions
    - Technical questions
    - Domain questions
    - Conceptual questions

    Args:
        chunks: List of document chunks

    Returns:
        List of all QA pairs (flattened from all categories)
    """
    agent = create_enhanced_qa_agent()
    all_qa_pairs = []

    for chunk in chunks:
        # Create prompt with chunk context
        prompt = f"""Generate QA pairs from this cybersecurity/AI text:

CHUNK TEXT:
{chunk.text}

METADATA:
- Section: {chunk.section_title or 'N/A'}
- Tags: {', '.join(chunk.tags) if chunk.tags else 'N/A'}
- Entities: {', '.join(chunk.entities) if chunk.entities else 'N/A'}

Generate 5-10 high-quality QA pairs categorized by type (NER, Technical, Domain, Conceptual).
"""

        try:
            result = await agent.run(prompt)
            output = result.data

            # Flatten all QA pairs and add quality scores
            chunk_qa_pairs = []

            # Add NER questions
            for qa in output.ner_questions:
                qa.quality_score = 0.9  # NER questions are high quality
                qa.metadata = qa.metadata or {}
                qa.metadata["question_type"] = "ner"
                qa.metadata["chunk_id"] = chunk.id
                chunk_qa_pairs.append(qa)

            # Add technical questions
            for qa in output.technical_questions:
                qa.quality_score = 0.95  # Technical questions are highest quality
                qa.metadata = qa.metadata or {}
                qa.metadata["question_type"] = "technical"
                qa.metadata["chunk_id"] = chunk.id
                chunk_qa_pairs.append(qa)

            # Add domain questions
            for qa in output.domain_questions:
                qa.quality_score = 0.9
                qa.metadata = qa.metadata or {}
                qa.metadata["question_type"] = "domain"
                qa.metadata["chunk_id"] = chunk.id
                chunk_qa_pairs.append(qa)

            # Add conceptual questions
            for qa in output.conceptual_questions:
                qa.quality_score = 0.85
                qa.metadata = qa.metadata or {}
                qa.metadata["question_type"] = "conceptual"
                qa.metadata["chunk_id"] = chunk.id
                chunk_qa_pairs.append(qa)

            all_qa_pairs.extend(chunk_qa_pairs)

        except Exception as e:
            import logging
            logging.warning(f"Failed to generate QA for chunk {chunk.id}: {e}")
            continue

    return all_qa_pairs


# Synchronous wrapper
def generate_enhanced_qa_pairs_sync(chunks: List[Chunk]) -> List[QAPair]:
    """Synchronous wrapper for enhanced QA generation."""
    import asyncio
    return asyncio.run(generate_enhanced_qa_pairs(chunks))
