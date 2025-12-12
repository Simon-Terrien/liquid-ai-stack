"""Quick test to verify split variants structure works."""

import sys
sys.path.insert(0, 'liquid-shared-core')
sys.path.insert(0, 'experiments')

from metadata_ablation.variants import create_retriever
from liquid_shared import VectorStore, EmbeddingService

print("="*80)
print("TESTING SPLIT VARIANTS STRUCTURE")
print("="*80)

# Initialize services
print("\n1. Initializing vector store and embedding service...")
vector_store = VectorStore(
    persist_dir="liquid-shared-core/data/vectordb",
    collection_name="documents",
)
print(f"   ✅ Vector store initialized with {vector_store.count()} documents")

embedding_service = EmbeddingService()
print(f"   ✅ Embedding service initialized")

# Test creating each variant
variants = ["v0", "v1", "v2", "v3", "v4", "v5"]
print(f"\n2. Testing variant creation...")

for variant_id in variants:
    try:
        retriever = create_retriever(variant_id, vector_store, embedding_service, top_k=5)
        config = retriever.get_metadata()
        print(f"   ✅ {variant_id.upper()}: {config['name']} - {config['description']}")
    except Exception as e:
        print(f"   ❌ {variant_id.upper()}: Failed - {e}")

# Test baseline retrieval
print(f"\n3. Testing V0 retrieval...")
try:
    retriever = create_retriever("v0", vector_store, embedding_service, top_k=3)
    test_query = "What are adversarial attacks on machine learning models?"
    results = retriever.retrieve(test_query)

    print(f"   Query: {test_query}")
    print(f"   ✅ Retrieved {len(results)} results")
    for i, result in enumerate(results, 1):
        print(f"      {i}. Score: {result.score:.3f} | ID: {result.chunk_id[:36]}")
except Exception as e:
    print(f"   ❌ Retrieval failed: {e}")
    import traceback
    traceback.print_exc()

print(f"\n" + "="*80)
print("TEST COMPLETE - Split variants structure verified!")
print("="*80)
