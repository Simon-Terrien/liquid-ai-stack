# Metadata Ablation Study - Experimental Design

## 🎯 Research Question

**How do different metadata features (keywords, categories, taxonomies) impact RAG retrieval quality?**

## 📊 Experimental Design

### Variants to Test

We'll test 6 retrieval variants to isolate the impact of each metadata enhancement:

| Variant | Description | Metadata Used | Purpose |
|---------|-------------|---------------|---------|
| **V0: Baseline** | Original ETL (no enhancements) | tags, entities only | Baseline performance |
| **V1: +Keywords** | Add keywords to metadata | tags, entities, keywords | Measure keyword impact |
| **V2: +Categories** | Add categories to metadata | tags, entities, categories | Measure classification impact |
| **V3: +Taxonomy** | Add hierarchical taxonomy | tags, entities, taxonomy | Measure hierarchy impact |
| **V4: Keywords+Categories** | Combine keywords & categories | tags, entities, keywords, categories | Measure synergy |
| **V5: Full Enhanced** | All metadata features | tags, entities, keywords, categories, taxonomy | Maximum enhancement |

### Retrieval Strategies per Variant

For each variant, we'll test different retrieval approaches:

#### V0: Baseline
```python
# Dense retrieval only
vector_store.search(query, top_k=10)
```

#### V1: +Keywords
```python
# Dense + BM25 keyword matching
hybrid_search(
    query=query,
    vector_weight=0.7,
    bm25_weight=0.3,
    bm25_fields=["text", "keywords"]
)
```

#### V2: +Categories
```python
# Dense + category filtering
category = predict_query_category(query)
vector_store.search(
    query,
    filter={"category": category},
    top_k=10
)
```

#### V3: +Taxonomy
```python
# Dense + taxonomy-based expansion
taxonomy_terms = expand_query_with_taxonomy(query)
vector_store.search(
    expanded_query=f"{query} {' '.join(taxonomy_terms)}",
    top_k=10
)
```

#### V4: Keywords+Categories
```python
# Hybrid (dense + BM25) + category filtering
category = predict_query_category(query)
hybrid_search(
    query=query,
    filter={"category": category},
    vector_weight=0.7,
    bm25_weight=0.3
)
```

#### V5: Full Enhanced
```python
# All strategies combined with importance weighting
category = predict_query_category(query)
taxonomy_terms = expand_query_with_taxonomy(query)
results = hybrid_search(
    query=f"{query} {' '.join(taxonomy_terms)}",
    filter={"category": category},
    vector_weight=0.7,
    bm25_weight=0.3
)
# Rerank by importance
reranked = rerank_by_importance(results, weight=0.2)
```

## 📝 Test Queries

We'll use a diverse set of test queries covering different aspects:

### Query Set (20 queries)

**AI/ML Security (5 queries)**:
1. "What are adversarial attacks on machine learning models?"
2. "How to defend against model poisoning?"
3. "What is adversarial training?"
4. "Model robustness techniques"
5. "Security risks in production AI systems"

**Data Protection / GDPR (5 queries)**:
6. "GDPR requirements for AI systems"
7. "Data protection measures for machine learning"
8. "Privacy regulations for AI"
9. "Personal data handling in AI"
10. "GDPR compliance requirements"

**Cyber Insurance (3 queries)**:
11. "Cyber insurance for AI systems"
12. "Risk assessment for AI insurance"
13. "Insurance coverage for AI incidents"

**Technical Controls (4 queries)**:
14. "Input validation for ML models"
15. "Runtime monitoring for AI systems"
16. "Anomaly detection in production"
17. "Model ensemble techniques"

**General Cybersecurity (3 queries)**:
18. "Threat intelligence for AI"
19. "Incident response for AI failures"
20. "Security audits for AI systems"

## 📏 Evaluation Metrics

### Primary Metrics

**1. Recall@K** (K ∈ {1, 3, 5, 10})
- Measures: How many relevant docs are retrieved
- Higher is better
- Most important metric for RAG

**2. Precision@K** (K ∈ {1, 3, 5, 10})
- Measures: Relevance of retrieved docs
- Higher is better
- Important for answer quality

**3. Mean Reciprocal Rank (MRR)**
- Measures: Position of first relevant result
- Higher is better
- Important for top result quality

**4. NDCG@K** (K ∈ {5, 10})
- Measures: Ranking quality
- Higher is better
- Considers graded relevance

### Secondary Metrics

**5. Category Prediction Accuracy**
- For V2, V4, V5: How often is predicted category correct?
- Measured against ground truth labels

**6. Query Latency**
- Measure retrieval time for each variant
- Trade-off: quality vs speed

**7. False Positive Rate**
- How many irrelevant docs in top-K?
- Lower is better

## 🏷️ Relevance Judgments

We need ground truth relevance judgments for each query-document pair.

### Relevance Scale
- **2 (Highly Relevant)**: Directly answers the query
- **1 (Relevant)**: Contains related information
- **0 (Not Relevant)**: Unrelated to query

### Creation Method

**Option 1: Manual Annotation** (Recommended)
```python
# Create annotation tool
for query in test_queries:
    results = vector_store.search(query, top_k=20)
    for doc in results:
        relevance = annotate(query, doc)  # Human labels 0, 1, or 2
        judgments[query][doc.id] = relevance
```

**Option 2: LLM-Based Annotation** (Faster, less reliable)
```python
# Use LFM2-1.2B to judge relevance
for query in test_queries:
    results = vector_store.search(query, top_k=20)
    for doc in results:
        relevance = llm_judge_relevance(query, doc.text)
        judgments[query][doc.id] = relevance
```

**Option 3: Hybrid** (Best)
- LLM generates initial judgments
- Human validates edge cases (relevance = 1)

## 🔬 Experimental Procedure

### Phase 1: Data Preparation
1. ✅ Run enhanced ETL on all documents (in progress)
2. Create test query set
3. Generate relevance judgments (manual or LLM-based)
4. Split data: 80% queries for tuning, 20% for final evaluation

### Phase 2: Baseline Establishment
1. Run V0 (baseline) on all test queries
2. Compute all metrics
3. Establish baseline performance

### Phase 3: Ablation Testing
For each variant V1-V5:
1. Configure retrieval strategy
2. Run on all test queries
3. Compute all metrics
4. Compare to baseline

### Phase 4: Statistical Analysis
1. Paired t-test: Compare each variant to baseline
2. Effect size: Cohen's d for each metric
3. Confidence intervals (95%)
4. Significance threshold: p < 0.05

### Phase 5: Analysis & Insights
1. Which metadata feature has highest impact?
2. Do features combine synergistically (V4, V5 > V1 + V2)?
3. Is the improvement statistically significant?
4. What's the latency cost?

## 📊 Expected Results Format

```python
{
    "variant": "V5: Full Enhanced",
    "metrics": {
        "recall@1": 0.75,
        "recall@3": 0.90,
        "recall@5": 0.95,
        "recall@10": 0.98,
        "precision@1": 0.85,
        "precision@3": 0.78,
        "precision@5": 0.72,
        "precision@10": 0.65,
        "mrr": 0.82,
        "ndcg@5": 0.88,
        "ndcg@10": 0.86,
        "avg_latency_ms": 145,
        "category_accuracy": 0.88  # For V2, V4, V5
    },
    "vs_baseline": {
        "recall@5_improvement": "+18%",
        "precision@5_improvement": "+12%",
        "mrr_improvement": "+15%",
        "p_value": 0.003,
        "cohens_d": 0.85,
        "significant": true
    }
}
```

## 📈 Visualization Plan

### 1. Metric Comparison Chart
```python
# Bar chart comparing all variants
metrics = ["Recall@5", "Precision@5", "MRR", "NDCG@5"]
variants = ["V0", "V1", "V2", "V3", "V4", "V5"]
# Grouped bar chart showing all metrics
```

### 2. Ablation Heatmap
```python
# Heatmap showing metric improvements
# Rows: Metrics
# Cols: Variants
# Colors: % improvement over baseline
```

### 3. Statistical Significance Matrix
```python
# Matrix showing pairwise comparisons
# V0 vs V1, V0 vs V2, ..., V4 vs V5
# * for p < 0.05, ** for p < 0.01, *** for p < 0.001
```

### 4. Latency vs Quality Trade-off
```python
# Scatter plot
# X-axis: Avg latency (ms)
# Y-axis: Recall@5
# Each point: One variant
```

## 💻 Implementation Plan

### File Structure
```
experiments/
├── metadata_ablation/
│   ├── __init__.py
│   ├── config.py           # Experiment config
│   ├── queries.py          # Test query definitions
│   ├── relevance.py        # Relevance judgment tool
│   ├── variants.py         # Retrieval variant implementations
│   ├── evaluator.py        # Metric computation
│   ├── statistical.py      # Statistical tests
│   └── visualize.py        # Result visualization
├── data/
│   ├── test_queries.json
│   ├── relevance_judgments.json
│   └── results/
│       ├── v0_baseline.json
│       ├── v1_keywords.json
│       ├── ...
│       └── summary.json
└── run_ablation.py          # Main runner
```

### Usage
```bash
# Step 1: Create relevance judgments
python experiments/metadata_ablation/relevance.py \
    --queries experiments/data/test_queries.json \
    --output experiments/data/relevance_judgments.json \
    --method hybrid  # llm + human validation

# Step 2: Run ablation study
python experiments/run_ablation.py \
    --variants all \
    --queries experiments/data/test_queries.json \
    --judgments experiments/data/relevance_judgments.json \
    --output experiments/data/results/

# Step 3: Analyze results
python experiments/metadata_ablation/statistical.py \
    --results experiments/data/results/ \
    --output experiments/data/results/summary.json

# Step 4: Visualize
python experiments/metadata_ablation/visualize.py \
    --results experiments/data/results/summary.json \
    --output experiments/data/results/plots/
```

## 🎓 Research Contributions

### Claims We Can Make (If Results Support)

1. **Metadata Impact**: "Rich metadata improves RAG retrieval recall@5 by X% (p < 0.05)"

2. **Feature Importance**: "Keywords provide Y% improvement, while categories add Z%"

3. **Synergy**: "Combined metadata features show synergistic effects beyond additive gains"

4. **Practical Guidance**: "For latency-sensitive applications, keywords alone provide 80% of benefit at 50% cost"

### Paper Sections Enhanced

**Methods**:
- Detailed ablation methodology
- Statistical rigor (paired tests, effect sizes)

**Results**:
- Quantitative improvements for each feature
- Statistical significance tests
- Latency analysis

**Discussion**:
- Which metadata matters most
- When to use which features
- Cost-benefit analysis

## ⚡ Quick Start (After ETL Completes)

1. **Create test queries**: Edit `experiments/data/test_queries.json`
2. **Generate judgments**: Run `relevance.py` (manual or LLM)
3. **Run baseline**: `python run_ablation.py --variants v0`
4. **Run ablations**: `python run_ablation.py --variants all`
5. **Analyze**: Check `experiments/data/results/summary.json`
6. **Visualize**: Generate plots with `visualize.py`

## 🎯 Success Criteria

**Minimum Viable Result**:
- ✅ At least one variant shows statistically significant improvement (p < 0.05)
- ✅ Effect size > 0.3 (Cohen's d) for at least one metric
- ✅ Latency overhead < 2x baseline

**Strong Result**:
- ✅ Multiple variants show significant improvement
- ✅ Effect sizes > 0.5 for key metrics (Recall@5, MRR)
- ✅ Clear ranking of feature importance
- ✅ Synergistic effects demonstrated

**Publication-Grade Result**:
- ✅ All enhanced variants beat baseline (p < 0.01)
- ✅ Effect sizes > 0.8 for V5 (Full Enhanced)
- ✅ Comprehensive statistical analysis
- ✅ Replicable methodology
- ✅ Clear practical recommendations

---

**Status**: Design complete, ready to implement after ETL finishes
**Next**: Implement relevance judgment tool and variant strategies
