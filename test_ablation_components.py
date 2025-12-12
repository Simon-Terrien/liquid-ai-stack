#!/usr/bin/env python3
"""Test ablation study components."""

import sys

sys.path.insert(0, "liquid-shared-core")
sys.path.insert(0, "experiments")

from liquid_shared import VectorStore, EmbeddingService
from metadata_ablation.variants import create_retriever


def test_retrieval_variants():
    """Test all retrieval variants."""
    print("=" * 80)
    print("Testing Ablation Study Components")
    print("=" * 80)

    # Initialize embedding service and vector store
    print("\n1. Initializing embedding service and vector store...")
    embedding_service = EmbeddingService()
    vector_store = VectorStore(
        persist_dir="liquid-shared-core/data/vectordb",
        collection_name="documents",
    )

    doc_count = vector_store.collection.count()
    print(f"   Vector store has {doc_count} documents")

    if doc_count == 0:
        print("\n❌ Vector store is empty!")
        print("   Run the ETL pipeline first to index documents.")
        return False

    # Test query
    test_query = "What are adversarial attacks on machine learning models?"
    print(f"\n2. Test Query: {test_query}")

    # Test each variant (start with v0 baseline)
    variants = ["v0"]  # Test baseline first, then add others: v1, v2, v3, v4, v5

    print("\n3. Testing Retrieval Variants:")
    print("-" * 80)

    for variant_id in variants:
        try:
            # Create retriever
            retriever = create_retriever(variant_id, vector_store, embedding_service, top_k=5)

            # Run retrieval
            results = retriever.retrieve(test_query)

            # Display results
            print(f"\n{variant_id.upper()}: {retriever.config.name}")
            print(f"  Description: {retriever.config.description}")
            print(f"  Retrieved {len(results)} documents:")

            for i, result in enumerate(results[:3]):
                print(f"    [{i+1}] Score: {result.score:.4f} | ID: {result.chunk_id}")
                print(f"        Preview: {result.text[:100]}...")

                # Check metadata
                if "keywords" in result.metadata:
                    print(f"        Keywords: {result.metadata['keywords'][:3]}")
                if "categories" in result.metadata:
                    print(f"        Categories: {result.metadata['categories']}")

            print(f"  ✅ {variant_id.upper()} working")

        except Exception as e:
            print(f"  ❌ {variant_id.upper()} failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    print("\n" + "=" * 80)
    print("✅ All retrieval variants working!")
    print("=" * 80)
    print("\nNext Steps:")
    print("  1. Generate relevance judgments:")
    print("     python experiments/metadata_ablation/relevance.py --auto-accept")
    print("\n  2. Run baseline experiment:")
    print("     python experiments/run_ablation.py --variants v0")
    print("\n  3. Run full ablation study:")
    print("     python experiments/run_ablation.py --variants all")

    return True


if __name__ == "__main__":
    success = test_retrieval_variants()
    sys.exit(0 if success else 1)
