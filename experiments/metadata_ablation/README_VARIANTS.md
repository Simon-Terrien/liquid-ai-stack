# Retrieval Variants Package

The `variants/` package contains modular implementations of 6 retrieval strategies (V0-V5) for the metadata ablation study.

## Quick Start

```python
from metadata_ablation.variants import create_retriever

# Create any variant (v0-v5)
retriever = create_retriever("v0", vector_store, embedding_service, top_k=10)

# Retrieve documents
results = retriever.retrieve("What are adversarial attacks?")
```

## Package Structure

```
variants/
├── __init__.py          # Package exports (40 lines)
├── base.py              # BaseRetriever + RetrieverConfig (100 lines)
├── factory.py           # create_retriever() (45 lines)
│
├── v0_baseline.py       # ✅ Dense only (33 lines)
├── v1_keywords.py       # ⏳ Dense + BM25 (43 lines)
├── v2_categories.py     # ⏳ Dense + filtering (82 lines)
├── v3_taxonomy.py       # ⏳ Dense + expansion (56 lines)
├── v4_hybrid.py         # ⏳ Keywords + Categories (60 lines)
└── v5_full_enhanced.py  # ⏳ All features (95 lines)
```

## Variants Overview

| Variant | Name | Features | Status |
|---------|------|----------|--------|
| V0 | Baseline | Dense retrieval only | ✅ Complete |
| V1 | +Keywords | Dense + BM25 keywords | ⏳ Needs BM25 |
| V2 | +Categories | Dense + category filtering | ⏳ Needs LLM |
| V3 | +Taxonomy | Dense + query expansion | ⏳ Needs graph |
| V4 | Hybrid | Keywords + Categories | ⏳ Needs V1+V2 |
| V5 | Full Enhanced | All features + importance | ⏳ Needs all |

## Usage Examples

### Basic Usage

```python
from metadata_ablation.variants import create_retriever
from liquid_shared import VectorStore, EmbeddingService

# Initialize services
vector_store = VectorStore(persist_dir="data/vectordb")
embedding_service = EmbeddingService()

# Create baseline retriever
retriever = create_retriever("v0", vector_store, embedding_service, top_k=5)

# Retrieve documents
results = retriever.retrieve("GDPR compliance requirements")

# Process results
for result in results:
    print(f"Score: {result.score:.3f} | Text: {result.text[:100]}")
```

### Advanced Usage - Direct Import

```python
from metadata_ablation.variants.v0_baseline import BaselineRetriever
from metadata_ablation.variants.v5_full_enhanced import FullEnhancedRetriever

# Create specific variants
baseline = BaselineRetriever(vector_store, embedding_service, top_k=10)
enhanced = FullEnhancedRetriever(
    vector_store,
    embedding_service,
    top_k=10,
    bm25_weight=0.3,
    importance_weight=0.2,
)

# Compare results
baseline_results = baseline.retrieve(query)
enhanced_results = enhanced.retrieve(query)
```

### Running All Variants

```python
from metadata_ablation.variants import create_retriever

variants = ["v0", "v1", "v2", "v3", "v4", "v5"]
query = "What are adversarial attacks on ML models?"

for variant_id in variants:
    retriever = create_retriever(variant_id, vector_store, embedding_service)
    results = retriever.retrieve(query)
    config = retriever.get_metadata()

    print(f"{config['name']}: {len(results)} results")
```

## Implementation Guide

### Creating a New Variant

1. Create file: `vX_variant_name.py`
2. Import base class:
   ```python
   from .base import BaseRetriever, RetrieverConfig
   ```
3. Implement retriever:
   ```python
   class NewRetriever(BaseRetriever):
       def __init__(self, vector_store, embedding_service, top_k=10):
           config = RetrieverConfig(
               name="VX: Name",
               description="Description",
               top_k=top_k,
           )
           super().__init__(vector_store, embedding_service, config)

       def retrieve(self, query: str) -> list[RetrievalResult]:
           # Implementation here
           pass
   ```
4. Update `factory.py` to add new variant
5. Update `__init__.py` to export new class

### Testing a Variant

```python
# Test imports
from metadata_ablation.variants import create_retriever

# Test creation
retriever = create_retriever("vX", vector_store, embedding_service)

# Test retrieval
results = retriever.retrieve("test query")
assert len(results) > 0

# Test metadata
metadata = retriever.get_metadata()
print(metadata)
```

## Variant Details

### V0: Baseline

**Purpose**: Establish baseline performance without enhanced metadata

**Implementation**: `v0_baseline.py`
```python
def retrieve(self, query: str) -> list[RetrievalResult]:
    query_embedding = self._query_to_embedding(query)
    results = self.vector_store.query(
        query_embedding=query_embedding,
        top_k=self.config.top_k
    )
    return self._results_to_retrieval_results(results)
```

**Status**: ✅ Complete and tested
- Recall@5: 50%
- Precision: 100%
- Latency: 23ms

### V1: +Keywords

**Purpose**: Add BM25 keyword matching on `metadata.keywords` field

**Implementation**: `v1_keywords.py`

**TODO**:
- [ ] Implement BM25 scoring
- [ ] Combine dense + sparse scores
- [ ] Add hybrid fusion (RRF or weighted)

**Expected Improvement**: +10-15% recall on keyword-heavy queries

### V2: +Categories

**Purpose**: Filter results by predicted query category

**Implementation**: `v2_categories.py`

**Current**:
- Rule-based category prediction (8 categories)
- Dense retrieval with post-filtering
- Fallback to unfiltered if insufficient results

**TODO**:
- [ ] Replace rule-based with LLM classification
- [ ] Add ChromaDB where clause filtering
- [ ] Measure precision improvement

**Expected Improvement**: +20% precision, slight recall trade-off

### V3: +Taxonomy

**Purpose**: Expand query with hierarchical taxonomy terms

**Implementation**: `v3_taxonomy.py`

**Current**:
- Simple synonym expansion
- Dense retrieval on expanded query

**TODO**:
- [ ] Extract taxonomies from metadata
- [ ] Implement graph traversal
- [ ] Add parent/child term expansion

**Expected Improvement**: +15% recall on underspecified queries

### V4: Hybrid (Keywords + Categories)

**Purpose**: Combine best of V1 and V2

**Implementation**: `v4_hybrid.py`

**Dependencies**: Requires V1 and V2 implementations

**TODO**:
- [ ] Implement V1 BM25 logic
- [ ] Implement V2 classification logic
- [ ] Combine hybrid search + filtering

**Expected Improvement**: Best precision-recall balance

### V5: Full Enhanced

**Purpose**: All metadata features combined

**Implementation**: `v5_full_enhanced.py`

**Features**:
- BM25 keywords (V1)
- Category filtering (V2)
- Taxonomy expansion (V3)
- Importance reranking (unique)

**Importance Reranking** (implemented):
```python
def _rerank_by_importance(self, results):
    for result in results:
        importance = result.metadata.get("importance", 5)
        normalized = importance / 10.0
        result.score = (
            result.score * (1 - self.config.importance_weight)
            + normalized * self.config.importance_weight
        )
    return sorted(results, key=lambda r: r.score, reverse=True)
```

**TODO**:
- [ ] Integrate V1, V2, V3 implementations
- [ ] Tune importance_weight parameter
- [ ] Optimize multi-stage pipeline

**Expected Improvement**: +25-30% overall quality

## Base Classes

### RetrieverConfig

Pydantic model for variant configuration:

```python
class RetrieverConfig(BaseModel):
    name: str                        # Variant name (e.g., "V0: Baseline")
    description: str                 # Short description
    top_k: int = 10                 # Number of results
    vector_weight: float = 0.7      # Dense retrieval weight
    bm25_weight: float = 0.3        # BM25 keyword weight
    importance_weight: float = 0.2  # Importance reranking weight
```

### BaseRetriever

Abstract base class with shared utilities:

**Methods**:
- `__init__(vector_store, embedding_service, config)`: Initialize
- `_query_to_embedding(query)`: Convert query to vector
- `_results_to_retrieval_results(results)`: Parse Chroma results
- `retrieve(query)`: **Abstract** - Implement in subclasses
- `get_metadata()`: Return variant info for logging

## Testing

Run the test suite:

```bash
# Test all variants instantiate
python test_split_variants.py

# Test V0 retrieval
python test_ablation_components.py

# Run ablation study
python experiments/run_ablation.py --variants v0
```

## Performance Tips

1. **Batch queries**: Create retriever once, reuse for multiple queries
2. **Cache embeddings**: EmbeddingService caches query embeddings
3. **Tune top_k**: Start with 10, increase if recall is low
4. **Monitor latency**: V5 will be slower due to multi-stage pipeline

## Troubleshooting

**Issue**: Import error `ModuleNotFoundError: No module named 'metadata_ablation'`
- **Fix**: Add `experiments/` to PYTHONPATH or run from project root

**Issue**: `TypeError: __init__() got an unexpected keyword argument`
- **Fix**: Ensure all variants have correct signature with `embedding_service`

**Issue**: Empty results
- **Fix**: Check vector store has documents: `vector_store.count()`

**Issue**: Low scores
- **Fix**: Verify embeddings are normalized: `embedding_service.encode()`

## Contributing

When adding new variants:

1. Follow naming convention: `vX_feature_name.py`
2. Inherit from `BaseRetriever`
3. Document expected improvements
4. Add tests
5. Update factory and `__init__.py`
6. Run ablation study to measure impact

## References

- Base implementation: `base.py`
- Factory pattern: `factory.py`
- Example usage: `test_split_variants.py`
- Ablation study: `../run_ablation.py`
- Documentation: `VARIANTS_SPLIT_SUMMARY.md`

---

**Status**: ✅ Package structure complete, V0 tested, V1-V5 ready for implementation
