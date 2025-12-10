# Experiment Framework for Research Validation

This module provides a comprehensive experiment framework for validating the liquid-ai-stack system for research publication.

## Overview

The experiment framework addresses the key validation requirements identified for publication:

1. **Dual-Pipeline Efficiency**: Compare unified ETL vs separate pipelines
2. **Multi-Model Ablation**: Test different model size combinations
3. **RAG Quality Evaluation**: Measure Recall@K, MRR, NDCG
4. **Fine-Tuning Data Quality**: Assess ROUGE, BLEU, perplexity
5. **Ablation Studies**: Test chunking strategies, metadata features, hybrid retrieval
6. **Statistical Significance**: Bootstrap tests, confidence intervals, effect sizes
7. **Human Evaluation**: Annotation interface and protocols

## Architecture

```
liquid_shared/experiments/
├── __init__.py                 # Main exports
├── config.py                   # Experiment configuration
├── tracker.py                  # MLflow + local tracking
├── stats.py                    # Statistical tests
├── utils.py                    # Utility functions
├── orchestrator.py             # Master experiment runner
└── evaluators/
    ├── rag_evaluator.py       # RAG metrics (Recall@K, MRR, NDCG)
    └── generation_evaluator.py # Generation metrics (ROUGE, BLEU)
```

## Quick Start

### Running All Experiments

```python
from liquid_shared.experiments.orchestrator import run_research_validation

# Run all experiments
results = run_research_validation(
    output_dir="data/experiments",
    experiments=["dual_pipeline", "rag_quality"],
    mlflow_tracking_uri=None,  # or "http://localhost:5000"
)
```

### Running Individual Experiments

#### 1. Dual-Pipeline Efficiency

```python
from etl_pipeline.experiments.dual_pipeline_experiment import run_dual_pipeline_experiment

results = run_dual_pipeline_experiment()
print(f"Time savings: {results['efficiency_gains']['time_saving_percent']:.2f}%")
print(f"GPU savings: {results['efficiency_gains']['gpu_saving_percent']:.2f}%")
```

#### 2. RAG Quality Evaluation

```python
from rag_runtime.experiments.rag_quality_experiment import (
    run_rag_quality_experiment,
    QueryExample,
)

# Define test queries
test_queries = [
    QueryExample(
        query="What is GDPR?",
        relevant_doc_ids={"doc1", "doc3"},
        expected_answer="General Data Protection Regulation..."
    ),
    # ... more queries
]

results = run_rag_quality_experiment(test_queries=test_queries)
print(f"Dense Recall@5: {results['results']['dense']['metrics']['recall_at_k']['recall@5']:.4f}")
print(f"Hybrid MRR: {results['results']['hybrid']['metrics']['mrr']:.4f}")
```

#### 3. Fine-Tuning Data Quality

```python
from liquid_shared.experiments.evaluators import GenerationEvaluator

evaluator = GenerationEvaluator(
    compute_rouge=True,
    compute_bleu=True,
    compute_perplexity=False,  # Requires reference model
)

# Evaluate generated QA pairs
predictions = ["Answer 1", "Answer 2", ...]
references = ["Reference 1", "Reference 2", ...]

metrics = evaluator.evaluate(predictions, references)
print(f"ROUGE-L: {metrics.rougeL_f:.4f}")
print(f"BLEU: {metrics.bleu:.4f}")
```

## Configuration

### Experiment Config

```python
from liquid_shared.experiments import (
    ExperimentConfig,
    ModelConfig,
    DatasetConfig,
    EvalConfig,
)

config = ExperimentConfig(
    name="my_experiment",
    description="Testing hypothesis X",
    experiment_type="rag_quality",

    # Model configuration
    model=ModelConfig(
        name="LFM2-700M",
        device="auto",
        dtype="auto",
        max_tokens=512,
    ),

    # Dataset configuration
    dataset=DatasetConfig(
        name="default",
        split="test",
        max_samples=100,
        seed=42,
    ),

    # Evaluation configuration
    eval=EvalConfig(
        compute_recall_at_k=True,
        recall_k_values=[1, 3, 5, 10, 20],
        compute_mrr=True,
        compute_ndcg=True,
        n_bootstrap_samples=1000,
        confidence_level=0.95,
    ),

    # Tracking
    mlflow_tracking_uri="http://localhost:5000",
    mlflow_experiment_name="liquid-rag-quality",
    n_runs=3,  # For statistical significance
    seed=42,
)
```

## Statistical Testing

The framework provides comprehensive statistical tests for comparing methods:

```python
from liquid_shared.experiments import StatisticalTests, compare_multiple_methods

# Compare multiple methods
results = {
    "baseline": [0.65, 0.68, 0.70, 0.67, 0.69],
    "method_a": [0.72, 0.75, 0.73, 0.74, 0.76],
    "method_b": [0.78, 0.80, 0.79, 0.81, 0.82],
}

comparisons = compare_multiple_methods(
    results,
    alpha=0.05,
    baseline_key="baseline"
)

# Each comparison includes:
# - Mean and confidence intervals
# - p-value and significance
# - Effect size (Cohen's d)
```

### Available Statistical Tests

1. **Paired t-test**: For comparing same items under two conditions
2. **Independent t-test**: For comparing different items
3. **Wilcoxon signed-rank**: Non-parametric paired test
4. **Mann-Whitney U**: Non-parametric independent test
5. **Bootstrap test**: Resampling-based significance test

## Evaluation Metrics

### RAG Metrics (Information Retrieval)

- **Recall@K**: Proportion of relevant documents in top-K results
- **Precision@K**: Proportion of top-K results that are relevant
- **MRR (Mean Reciprocal Rank)**: Average of reciprocal ranks of first relevant result
- **MAP (Mean Average Precision)**: Mean of average precisions across queries
- **NDCG@K**: Normalized Discounted Cumulative Gain at K

### Generation Metrics (Text Quality)

- **ROUGE-1/2/L**: N-gram overlap with reference (recall, precision, F1)
- **BLEU**: N-gram precision with brevity penalty
- **Exact Match**: Percentage of exact string matches
- **Token F1**: F1 score at token level
- **Perplexity**: Language model probability (optional)

## Experiment Tracking

Results are tracked both locally and optionally to MLflow:

### Local Tracking
- `output_dir/params.json` - Experiment parameters
- `output_dir/metrics.jsonl` - Time-series metrics
- `output_dir/summary.json` - Final summary
- `output_dir/experiment_report.md` - Human-readable report

### MLflow Tracking
- Automatic parameter and metric logging
- System info (GPU, memory, versions)
- Artifact storage (models, configs, plots)
- Experiment comparison UI

To start MLflow UI:
```bash
mlflow ui --backend-store-uri file:///path/to/mlruns
```

## Best Practices

### 1. Set Random Seeds
```python
from liquid_shared.experiments import set_seed
set_seed(42)  # For reproducibility
```

### 2. Run Multiple Trials
```python
config = ExperimentConfig(
    name="my_experiment",
    n_runs=5,  # Run 5 times for statistical significance
    seed=42,
)
```

### 3. Log System Info
```python
from liquid_shared.experiments import log_system_info
system_info = log_system_info()  # Logs GPU, CPU, versions
```

### 4. Use Confidence Intervals
```python
from liquid_shared.experiments import compute_confidence_interval

mean, lower, upper = compute_confidence_interval(
    data=[0.65, 0.68, 0.70, 0.67, 0.69],
    confidence=0.95,
    method="bootstrap",
)
print(f"Mean: {mean:.4f} [{lower:.4f}, {upper:.4f}]")
```

## Example: Complete Experiment Workflow

```python
import logging
from pathlib import Path
from liquid_shared.experiments import (
    ExperimentConfig,
    ExperimentTracker,
    ModelConfig,
    DatasetConfig,
    EvalConfig,
    set_seed,
    log_system_info,
)
from liquid_shared.experiments.evaluators import RAGEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO)

# 1. Define experiment config
config = ExperimentConfig(
    name="rag_quality_ablation",
    description="Test impact of metadata boosting on retrieval",
    experiment_type="metadata_ablation",
    model=ModelConfig(name="LFM2-700M"),
    dataset=DatasetConfig(name="default", max_samples=100),
    eval=EvalConfig(recall_k_values=[1, 3, 5, 10]),
    n_runs=3,
    seed=42,
)

# 2. Setup tracker
output_dir = Path("data/experiments/metadata_ablation")
tracker = ExperimentTracker(
    experiment_name="rag-metadata-ablation",
    run_name="test_run_1",
    output_dir=output_dir,
)

# 3. Run experiment
with tracker:
    # Log config and system info
    tracker.log_params(config.to_dict())
    tracker.log_params(log_system_info())
    set_seed(config.seed)

    # Run your experiment logic
    for variant in ["baseline", "with_metadata"]:
        # ... retrieve results ...
        # ... compute metrics ...

        # Log metrics
        tracker.log_metrics({
            f"{variant}/recall@5": 0.75,
            f"{variant}/mrr": 0.82,
        })

    # Statistical comparison
    # ... compute significance tests ...
    tracker.log_dict(stat_results, "statistical_tests.json")

# 4. Results are automatically saved to:
# - output_dir/summary.json
# - output_dir/metrics.jsonl
# - output_dir/params.json
# - MLflow (if configured)
```

## Future Enhancements

The following experiments are planned but not yet fully implemented:

1. **Multi-Model Ablation**: Test all combinations of model sizes for different tasks
2. **Chunking Strategy Ablation**: Compare fixed-size, semantic, and recursive chunking
3. **Metadata Feature Importance**: Measure contribution of each metadata field
4. **Hybrid Weight Tuning**: Optimize dense/sparse weights in hybrid retrieval
5. **QA Validation Threshold**: Find optimal quality threshold for filtering
6. **Human Evaluation Interface**: Web UI for collecting human judgments
7. **Cost Analysis**: Track inference costs and cost/quality tradeoffs

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError`, ensure dependencies are installed:

```bash
cd liquid-shared-core
pip install -e .
# Or with uv:
uv sync
```

### MLflow Connection Issues

If MLflow tracking fails, it falls back to local-only tracking. To debug:

```python
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("test")  # Should not error if server is running
```

### Memory Issues

For large experiments, adjust batch sizes or use CPU mode:

```python
config = ModelConfig(
    name="LFM2-700M",
    device="cpu",  # Force CPU mode
    dtype="fp32",
)
```

## Contributing

When adding new experiments:

1. Create experiment class in appropriate package
2. Implement `run()` method returning dict with results
3. Add statistical tests comparing to baselines
4. Log all metrics to tracker
5. Add entry in orchestrator.py
6. Update this README with usage examples

## References

- MLflow: https://mlflow.org/docs/latest/index.html
- ROUGE: https://github.com/google-research/google-research/tree/master/rouge
- Statistical Testing: https://docs.scipy.org/doc/scipy/reference/stats.html
