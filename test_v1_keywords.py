"""Test V1 Keywords retriever with BM25."""

import sys
sys.path.insert(0, 'liquid-shared-core')
sys.path.insert(0, 'experiments')

from metadata_ablation.variants import create_retriever
from liquid_shared import VectorStore, EmbeddingService

print("="*80)
print("TESTING V1: KEYWORDS (Dense + BM25)")
print("="*80)

# Initialize services
vector_store = VectorStore(persist_dir="liquid-shared-core/data/vectordb")
embedding_service = EmbeddingService()

print(f"\n✅ Vector store: {vector_store.count()} documents")

# Create V1 retriever
retriever = create_retriever("v1", vector_store, embedding_service, top_k=5)
print(f"✅ V1 retriever created: {retriever.config.name}")
print(f"   BM25 weight: {retriever.config.bm25_weight}")

# Test queries
test_queries = [
    "What are adversarial attacks on machine learning models?",
    "GDPR compliance requirements",
    "Cyber insurance risk assessment",
]

for i, query in enumerate(test_queries, 1):
    print(f"\n{'-'*80}")
    print(f"Query {i}: {query}")
    print(f"{'-'*80}")

    try:
        results = retriever.retrieve(query)
        print(f"✅ Retrieved {len(results)} results\n")

        for j, result in enumerate(results, 1):
            # Extract keywords for display
            tags = result.metadata.get('tags', '')
            tags_list = [t.strip() for t in tags.split(',')[:3]] if tags else []
            tags_display = ', '.join(tags_list) if tags_list else 'N/A'

            print(f"  {j}. Score: {result.score:.4f}")
            print(f"     ID: {result.chunk_id[:36]}")
            print(f"     Tags: {tags_display}")
            print(f"     Text: {result.text[:100]}...")
            print()

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

print("="*80)
print("V1 KEYWORDS TEST COMPLETE")
print("="*80)
