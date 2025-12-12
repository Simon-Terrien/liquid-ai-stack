#!/usr/bin/env python3
"""Simple relevance judgment generator - uses direct LLM calls without Pydantic AI."""

import json
import sys
from pathlib import Path

sys.path.insert(0, "liquid-shared-core")

from liquid_shared import VectorStore, EmbeddingService


def generate_simple_judgments(
    vector_store: VectorStore,
    embedding_service: EmbeddingService,
    queries_path: Path,
    output_path: Path,
    top_k: int = 10,
):
    """Generate simple relevance judgments using a heuristic approach.

    For MVP, we'll use a simple heuristic:
    - Docs with score > 0.75: Highly Relevant (2)
    - Docs with score > 0.5: Relevant (1)
    - Docs with score <= 0.5: Not Relevant (0)

    This allows us to test the framework without needing LLM-based judgments.
    """
    print("=" * 80)
    print("Simple Relevance Judgment Generation")
    print("=" * 80)
    print("\nUsing heuristic-based judgments (score-based)")
    print("  - Score > 0.75: Highly Relevant (2)")
    print("  - Score > 0.5: Relevant (1)")
    print("  - Score <= 0.5: Not Relevant (0)\n")

    # Load queries
    with open(queries_path) as f:
        data = json.load(f)
        queries = data["queries"]

    print(f"Loaded {len(queries)} test queries\n")

    all_judgments = {}

    for query in queries:
        query_id = query["id"]
        query_text = query["text"]

        print(f"\nProcessing query {query_id}: {query_text}")

        # Get embedding
        query_embedding = embedding_service.encode([query_text])
        if hasattr(query_embedding[0], 'tolist'):
            query_embedding = query_embedding[0].tolist()
        else:
            query_embedding = query_embedding[0]

        # Retrieve documents
        chroma_results = vector_store.query(query_embedding=query_embedding, top_k=top_k)

        # Parse results
        ids = chroma_results.get("ids", [[]])[0]
        documents = chroma_results.get("documents", [[]])[0]
        distances = chroma_results.get("distances", [[]])[0]

        # Generate judgments based on similarity scores
        judgments = []
        for i, doc_id in enumerate(ids):
            distance = distances[i] if i < len(distances) else 1.0
            # Convert distance to similarity (cosine distance in [0, 2])
            score = 1 - (distance / 2.0)

            # Heuristic-based relevance
            if score > 0.75:
                relevance = 2  # Highly relevant
                reasoning = f"High similarity score ({score:.3f})"
            elif score > 0.5:
                relevance = 1  # Relevant
                reasoning = f"Medium similarity score ({score:.3f})"
            else:
                relevance = 0  # Not relevant
                reasoning = f"Low similarity score ({score:.3f})"

            judgments.append({
                "doc_id": doc_id,
                "relevance": relevance,
                "reasoning": reasoning,
                "key_matches": [],
                "source": "heuristic",
                "score": score
            })

        # Count relevance levels
        counts = {0: 0, 1: 0, 2: 0}
        for j in judgments:
            counts[j["relevance"]] += 1

        print(f"  Retrieved: {len(judgments)} docs")
        print(f"  Relevance: 2 (Highly)={counts[2]}, 1 (Relevant)={counts[1]}, 0 (Not)={counts[0]}")

        all_judgments[query_id] = {
            "query": query_text,
            "category": query["category"],
            "expected_topics": query["expected_topics"],
            "judgments": judgments
        }

        # Save progress after each query
        with open(output_path, "w") as f:
            json.dump(all_judgments, f, indent=2)

    print(f"\n{'='*80}")
    print(f"✅ All judgments complete!")
    print(f"Saved {len(all_judgments)} queries to {output_path}")
    print(f"{'='*80}")

    return all_judgments


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate simple heuristic-based relevance judgments")
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
        "--top-k",
        type=int,
        default=10,
        help="Number of documents to judge per query",
    )
    args = parser.parse_args()

    # Initialize services
    print("Initializing embedding service and vector store...")
    embedding_service = EmbeddingService()
    vector_store = VectorStore(
        persist_dir="liquid-shared-core/data/vectordb",
        collection_name="documents",
    )

    print(f"Vector store has {vector_store.collection.count()} documents\n")

    # Generate judgments
    generate_simple_judgments(
        vector_store=vector_store,
        embedding_service=embedding_service,
        queries_path=Path(args.queries),
        output_path=Path(args.output),
        top_k=args.top_k,
    )
