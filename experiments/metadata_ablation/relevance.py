"""Relevance judgment tool for creating ground truth labels."""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

sys.path.insert(0, "liquid-shared-core")
sys.path.insert(0, "liquid-etl-pipeline")

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from liquid_shared import VectorStore, EmbeddingService


class RelevanceOutput(BaseModel):
    """Structured output for relevance judgment."""

    relevance: Literal[0, 1, 2] = Field(
        description="0=Not Relevant, 1=Relevant, 2=Highly Relevant"
    )
    reasoning: str = Field(description="Explanation for the relevance score")
    key_matches: list[str] = Field(
        default_factory=list,
        description="Key concepts that match between query and document",
    )


@dataclass
class RelevanceJudgment:
    """A single relevance judgment."""

    query_id: str
    doc_id: str
    relevance: int  # 0, 1, or 2
    reasoning: str
    key_matches: list[str]
    source: str  # "llm", "human", or "hybrid"


class RelevanceJudger:
    """
    Tool for creating relevance judgments.

    Uses hybrid approach:
    1. LLM generates initial judgments
    2. Human validates edge cases (relevance = 1)
    """

    def __init__(self, model: str = "LiquidAI/LFM2-1.2B", auto_accept: bool = False):
        """Initialize relevance judger.

        Args:
            model: LLM model to use for judgments
            auto_accept: If True, skip human validation (LLM-only mode)
        """
        self.model = model
        self.auto_accept = auto_accept
        self.agent = Agent(
            f"outlines-transformers:{model}",
            result_type=RelevanceOutput,
            system_prompt=self._get_system_prompt(),
        )

    def _get_system_prompt(self) -> str:
        """Get system prompt for relevance judgment."""
        return """You are a relevance judgment expert for information retrieval evaluation.

Your task is to assess whether a document is relevant to a given query.

**Relevance Scale**:
- **2 (Highly Relevant)**: The document directly answers the query or provides key information needed
- **1 (Relevant)**: The document contains related information but doesn't fully answer the query
- **0 (Not Relevant)**: The document is unrelated to the query

**Guidelines**:
- Focus on semantic relevance, not keyword matching
- Consider whether the document would help answer the query
- Be strict: most documents should be 0 or 2, use 1 for edge cases
- Identify specific concepts that match between query and document

**Output**:
- Provide a relevance score (0, 1, or 2)
- Explain your reasoning clearly
- List key matching concepts
"""

    async def judge_async(self, query: str, doc_text: str) -> RelevanceOutput:
        """Generate LLM-based relevance judgment asynchronously.

        Args:
            query: The search query
            doc_text: The document text to judge

        Returns:
            Relevance judgment with score, reasoning, and key matches
        """
        prompt = f"""Query: {query}

Document:
{doc_text[:2000]}  # Truncate long documents

Assess the relevance of this document to the query."""

        result = await self.agent.run(prompt)
        return result.data

    def judge_sync(self, query: str, doc_text: str) -> RelevanceOutput:
        """Generate LLM-based relevance judgment synchronously."""
        import asyncio

        return asyncio.run(self.judge_async(query, doc_text))

    def validate_judgment(self, judgment: RelevanceOutput, query: str, doc_text: str) -> RelevanceJudgment:
        """Validate LLM judgment with human if needed.

        Args:
            judgment: LLM-generated judgment
            query: The query
            doc_text: The document text

        Returns:
            Validated relevance judgment
        """
        # Auto-accept if configured
        if self.auto_accept:
            return RelevanceJudgment(
                query_id="",
                doc_id="",
                relevance=judgment.relevance,
                reasoning=judgment.reasoning,
                key_matches=judgment.key_matches,
                source="llm",
            )

        # Auto-accept clear cases (0 or 2)
        if judgment.relevance in [0, 2]:
            return RelevanceJudgment(
                query_id="",
                doc_id="",
                relevance=judgment.relevance,
                reasoning=judgment.reasoning,
                key_matches=judgment.key_matches,
                source="llm",
            )

        # Human validation for edge cases (relevance = 1)
        print("\n" + "=" * 80)
        print("EDGE CASE - HUMAN VALIDATION NEEDED")
        print("=" * 80)
        print(f"\nQuery: {query}")
        print(f"\nDocument: {doc_text[:500]}...")
        print(f"\nLLM Judgment: {judgment.relevance}")
        print(f"Reasoning: {judgment.reasoning}")
        print(f"Key Matches: {', '.join(judgment.key_matches)}")
        print("\n" + "-" * 80)
        print("Options:")
        print("  0 - Not Relevant")
        print("  1 - Relevant (accept LLM judgment)")
        print("  2 - Highly Relevant")
        print("  q - Quit")
        print("-" * 80)

        while True:
            choice = input("\nYour judgment [0/1/2/q]: ").strip().lower()
            if choice == "q":
                raise KeyboardInterrupt("User quit validation")
            if choice in ["0", "1", "2"]:
                final_relevance = int(choice)
                source = "hybrid" if choice != str(judgment.relevance) else "llm"
                return RelevanceJudgment(
                    query_id="",
                    doc_id="",
                    relevance=final_relevance,
                    reasoning=judgment.reasoning,
                    key_matches=judgment.key_matches,
                    source=source,
                )
            print("Invalid choice. Please enter 0, 1, 2, or q.")

    def generate_judgments_for_query(
        self,
        query_id: str,
        query_text: str,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
        top_k: int = 20,
    ) -> list[RelevanceJudgment]:
        """Generate relevance judgments for all retrieved documents.

        Args:
            query_id: Query identifier
            query_text: The search query
            vector_store: Vector store to search
            embedding_service: Service for generating embeddings
            top_k: Number of documents to judge

        Returns:
            List of relevance judgments
        """
        # Retrieve candidate documents
        query_embedding = embedding_service.encode([query_text])
        if hasattr(query_embedding[0], 'tolist'):
            query_embedding = query_embedding[0].tolist()
        else:
            query_embedding = query_embedding[0]

        chroma_results = vector_store.query(query_embedding=query_embedding, top_k=top_k)

        # Convert to simple result objects
        from dataclasses import dataclass
        @dataclass
        class SimpleResult:
            chunk_id: str
            text: str

        results = []
        ids = chroma_results.get("ids", [[]])[0]
        documents = chroma_results.get("documents", [[]])[0]
        for i, chunk_id in enumerate(ids):
            results.append(SimpleResult(
                chunk_id=chunk_id,
                text=documents[i] if i < len(documents) else ""
            ))

        judgments = []
        for i, result in enumerate(results):
            print(f"\nJudging document {i+1}/{len(results)} for query: {query_text}")

            # Get LLM judgment
            llm_judgment = self.judge_sync(query_text, result.text)

            # Validate if needed
            validated = self.validate_judgment(llm_judgment, query_text, result.text)
            validated.query_id = query_id
            validated.doc_id = result.chunk_id

            judgments.append(validated)

        return judgments


def create_relevance_judgments(
    queries_path: Path,
    vector_store: VectorStore,
    embedding_service: EmbeddingService,
    output_path: Path,
    model: str = "LiquidAI/LFM2-1.2B",
    top_k: int = 20,
    auto_accept: bool = False,
):
    """Create relevance judgments for all test queries.

    Args:
        queries_path: Path to test queries JSON
        vector_store: Vector store with indexed documents
        embedding_service: Service for generating embeddings
        output_path: Path to save judgments
        model: LLM model to use
        top_k: Number of documents to judge per query
        auto_accept: If True, skip human validation
    """
    # Load queries
    with open(queries_path) as f:
        data = json.load(f)
        queries = data["queries"]

    # Initialize judger
    judger = RelevanceJudger(model=model, auto_accept=auto_accept)

    # Generate judgments for each query
    all_judgments = {}
    try:
        for query in queries:
            print(f"\n{'='*80}")
            print(f"Processing query {query['id']}: {query['text']}")
            print(f"{'='*80}")

            judgments = judger.generate_judgments_for_query(
                query_id=query["id"],
                query_text=query["text"],
                vector_store=vector_store,
                embedding_service=embedding_service,
                top_k=top_k,
            )

            all_judgments[query["id"]] = {
                "query": query["text"],
                "category": query["category"],
                "expected_topics": query["expected_topics"],
                "judgments": [
                    {
                        "doc_id": j.doc_id,
                        "relevance": j.relevance,
                        "reasoning": j.reasoning,
                        "key_matches": j.key_matches,
                        "source": j.source,
                    }
                    for j in judgments
                ],
            }

            # Save progress after each query
            with open(output_path, "w") as f:
                json.dump(all_judgments, f, indent=2)

            print(f"\n✅ Completed {query['id']} ({len(judgments)} judgments)")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Saving partial results...")
        with open(output_path, "w") as f:
            json.dump(all_judgments, f, indent=2)
        print(f"Saved {len(all_judgments)} queries to {output_path}")
        raise

    print(f"\n✅ All judgments complete! Saved to {output_path}")
    return all_judgments


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate relevance judgments")
    parser.add_argument(
        "--queries",
        default="experiments/data/test_queries.json",
        help="Path to test queries JSON",
    )
    parser.add_argument(
        "--output",
        default="experiments/data/relevance_judgments.json",
        help="Path to save judgments",
    )
    parser.add_argument(
        "--model",
        default="LiquidAI/LFM2-1.2B",
        help="LLM model to use",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of documents to judge per query",
    )
    parser.add_argument(
        "--auto-accept",
        action="store_true",
        help="Skip human validation (LLM-only mode)",
    )
    args = parser.parse_args()

    # Initialize embedding service and vector store
    embedding_service = EmbeddingService()
    vector_store = VectorStore(
        persist_dir="liquid-shared-core/data/vectordb",
        collection_name="documents",
    )

    # Create judgments
    create_relevance_judgments(
        queries_path=Path(args.queries),
        vector_store=vector_store,
        embedding_service=embedding_service,
        output_path=Path(args.output),
        model=args.model,
        top_k=args.top_k,
        auto_accept=args.auto_accept,
    )
