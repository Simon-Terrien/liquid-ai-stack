# Metadata Ablation Study

This directory contains the implementation of a comprehensive ablation study to measure the impact of enhanced metadata features on RAG retrieval quality.

## Overview

The ablation study evaluates **6 retrieval variants** to isolate the contribution of each metadata feature:

| Variant | Features | Description |
|---------|----------|-------------|
| **V0** | Baseline | Dense retrieval only (tags + entities) |
| **V1** | +Keywords | Dense + BM25 keyword matching |
| **V2** | +Categories | Dense + category filtering |
| **V3** | +Taxonomy | Dense + taxonomy-based expansion |
| **V4** | Keywords+Categories | Hybrid search + category filtering |
| **V5** | Full Enhanced | All features + importance weighting |

## Directory Structure

```
experiments/
├── README.md                      # This file
├── METADATA_ABLATION_STUDY.md     # Detailed experimental design
├── run_ablation.py                # Main experiment runner
├── metadata_ablation/             # Ablation study package
│   ├── __init__.py
│   ├── config.py                  # Configuration
│   ├── relevance.py               # Relevance judgment tool
│   ├── variants.py                # Retrieval variant implementations
│   ├── evaluator.py               # Metrics evaluation
│   ├── statistical.py             # Statistical analysis
│   └── visualize.py               # Visualization tools
└── data/
    ├── test_queries.json          # 20 test queries
    ├── relevance_judgments.json   # Ground truth labels (generated)
    └── results/                   # Experiment results
        ├── v0_results.json
        ├── v1_results.json
        ├── ...
        ├── summary.json
        └── plots/                 # Visualization outputs

## Workflow

### Step 1: Generate Relevance Judgments

Before running the ablation study, you need ground truth relevance labels for the test queries.

**Option A: LLM-only (fast, less accurate)**
```bash
python experiments/metadata_ablation/relevance.py \
    --queries experiments/data/test_queries.json \
    --output experiments/data/relevance_judgments.json \
    --auto-accept
```

**Option B: Hybrid LLM + Human (recommended, more accurate)**
```bash
python experiments/metadata_ablation/relevance.py \
    --queries experiments/data/test_queries.json \
    --output experiments/data/relevance_judgments.json
```

This will:
1. Retrieve top-20 documents for each query
2. Use LLM to generate initial relevance judgments (0, 1, or 2)
3. Prompt for human validation of edge cases (relevance = 1)
4. Save judgments to JSON file

**Progress is saved after each query**, so you can interrupt and resume later.

### Step 2: Run Baseline Experiment

Test the baseline (V0) variant first:

```bash
python experiments/run_ablation.py --variants v0
```

This will:
- Load test queries and relevance judgments
- Initialize vector store (must contain indexed documents from ETL)
- Run retrieval for all queries using V0 (baseline)
- Compute metrics (Recall@K, Precision@K, MRR, NDCG@K)
- Save results to `experiments/data/results/v0_results.json`

### Step 3: Run Full Ablation Study

Run all variants (V0-V5):

```bash
python experiments/run_ablation.py --variants all
```

Or run specific variants:

```bash
python experiments/run_ablation.py --variants v0 v1 v5
```

Results are saved incrementally, so you can run variants separately:

```bash
# Run sequentially
python experiments/run_ablation.py --variants v0
python experiments/run_ablation.py --variants v1
python experiments/run_ablation.py --variants v2
# ...
```

### Step 4: Generate Visualizations

After running all variants, generate plots:

```bash
python experiments/metadata_ablation/visualize.py \
    --results experiments/data/results/summary.json \
    --output experiments/data/results/plots
```

This creates:
- `metric_comparison.png` - Grouped bar chart comparing all metrics
- `ablation_heatmap.png` - Heatmap of % improvements over baseline
- `latency_vs_quality.png` - Scatter plot of latency vs Recall@5

### Step 5: Statistical Analysis

Statistical comparisons are included in the summary output, showing:
- Mean values for each variant
- % improvement over baseline
- p-values (significance testing)
- Cohen's d effect sizes
- 95% confidence intervals

## Evaluation Metrics

### Primary Metrics

**Recall@K** - Fraction of relevant documents retrieved in top-K
```
Recall@K = (# relevant in top-K) / (total # relevant)
```

**Precision@K** - Fraction of top-K that are relevant
```
Precision@K = (# relevant in top-K) / K
```

**MRR (Mean Reciprocal Rank)** - Reciprocal of first relevant result's rank
```
MRR = 1 / (rank of first relevant document)
```

**NDCG@K** - Normalized Discounted Cumulative Gain (considers graded relevance)
```
NDCG@K = DCG@K / IDCG@K
```

### Secondary Metrics

- **Avg Latency (ms)** - Average retrieval time per query
- **Category Accuracy** - For V2, V4, V5: accuracy of predicted query categories

## Output Files

### Per-Variant Results (`v0_results.json`, etc.)

```json
{
  "variant": "v0",
  "metrics": {
    "recall@1": 0.45,
    "recall@5": 0.78,
    "precision@1": 0.85,
    "precision@5": 0.62,
    "mrr": 0.72,
    "ndcg@5": 0.81,
    "avg_latency_ms": 125.3
  },
  "results": {
    "results_by_query": {
      "q001": ["doc_1", "doc_5", ...],
      ...
    },
    "latencies_by_query": {
      "q001": 120.5,
      ...
    }
  }
}
```

### Summary (`summary.json`)

```json
{
  "config": {
    "variants": ["v0", "v1", "v2", "v3", "v4", "v5"],
    "test_queries_count": 20,
    "top_k": 10
  },
  "results_by_variant": {
    "v0": { "recall@1": 0.45, ... },
    "v1": { "recall@1": 0.52, ... },
    ...
  }
}
```

### Relevance Judgments (`relevance_judgments.json`)

```json
{
  "q001": {
    "query": "What are adversarial attacks on machine learning models?",
    "category": "AI/ML Security",
    "expected_topics": ["adversarial attacks", "model security"],
    "judgments": [
      {
        "doc_id": "chunk_123",
        "relevance": 2,
        "reasoning": "Directly explains adversarial attacks",
        "key_matches": ["adversarial", "ML models", "perturbations"],
        "source": "llm"
      },
      ...
    ]
  },
  ...
}
```

## Expected Results

Based on the experimental design, we expect:

### Minimum Viable Result
- ✅ At least one variant shows significant improvement (p < 0.05)
- ✅ Effect size > 0.3 (Cohen's d) for at least one metric
- ✅ Latency overhead < 2x baseline

### Strong Result
- ✅ Multiple variants show significant improvement
- ✅ Effect sizes > 0.5 for key metrics (Recall@5, MRR)
- ✅ Clear ranking of feature importance
- ✅ Synergistic effects (V5 > V1 + V2 + V3)

### Publication-Grade Result
- ✅ All enhanced variants beat baseline (p < 0.01)
- ✅ Effect sizes > 0.8 for V5
- ✅ Comprehensive statistical analysis
- ✅ Replicable methodology

## Research Claims

If results support, we can claim:

1. **"Rich metadata improves RAG retrieval recall@5 by X% (p < 0.05)"**
   - Quantify improvement with statistical significance

2. **"Keywords provide Y% improvement, categories add Z%"**
   - Isolate individual feature contributions

3. **"Combined features show synergistic effects"**
   - V5 > V1 + V2 + V3 (more than additive gains)

4. **"For latency-sensitive applications, keywords alone provide 80% of benefit at 50% cost"**
   - Practical guidance for deployment

## Troubleshooting

### "Relevance judgments not found"
Run `relevance.py` first to generate ground truth labels.

### "Vector store has 0 documents"
Ensure ETL pipeline has run and indexed documents. Check:
```bash
ls -lah liquid-shared-core/data/vectordb/
```

### "No judgments for query qXXX"
The relevance judgment file may be incomplete. Re-run `relevance.py` for missing queries.

### Statistical tests not running
Ensure you have `scipy` and `numpy` installed:
```bash
pip install scipy numpy matplotlib seaborn
```

## Next Steps

After completing the ablation study:

1. **Analyze Results**
   - Compare metrics across variants
   - Identify which features have highest impact
   - Check for synergistic effects

2. **Write Research Paper Section**
   - Document methodology
   - Present results with visualizations
   - Discuss implications

3. **Implement Production Variant**
   - Choose best variant based on quality/latency trade-off
   - Integrate into RAG runtime

4. **Extend Study**
   - Test on larger query set
   - Evaluate on different domains
   - Fine-tune hyperparameters (weights, thresholds)

## References

- Ablation study design: `METADATA_ABLATION_STUDY.md`
- Test queries: `data/test_queries.json` (20 queries across 7 categories)
- Enhanced metadata implementation: `../ENHANCED_METADATA_COMPLETE.md`
