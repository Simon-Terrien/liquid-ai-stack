# Quick Start: Running Your First Experiments

This guide will walk you through running your first research validation experiments.

## Prerequisites

1. **Ensure the liquid-ai-stack is set up**:
   ```bash
   cd liquid-ai-stack
   ls data/raw/  # Should have some documents
   ls data/vectordb/  # Should have vector database from ETL
   ```

2. **Install dependencies**:
   ```bash
   cd liquid-shared-core
   pip install -e .
   # This installs numpy, scipy, rouge-score, and other experiment dependencies
   ```

3. **Verify installation**:
   ```bash
   python -c "from liquid_shared.experiments import ExperimentTracker; print('✓ Experiments module loaded')"
   python -c "from liquid_shared.experiments.evaluators import RAGEvaluator; print('✓ Evaluators loaded')"
   ```

## Option 1: Run a Quick Test (Recommended First)

Let's start with a small-scale RAG quality evaluation:

```bash
# From repository root
python -c "
from liquid_shared.experiments.orchestrator import run_research_validation

# Run just RAG quality with verbose logging
results = run_research_validation(
    output_dir='data/experiments/test',
    experiments=['rag_quality'],
)

print('\\n=== Test Complete ===')
print('Check results in: data/experiments/test/')
"
```

**What this does**:
- Loads your existing vector database
- Runs retrieval quality evaluation
- Compares dense vs hybrid retrieval
- Saves results to `data/experiments/test/`

**Expected output**:
```
Starting experiment orchestration: ['rag_quality']
Running experiment: rag_quality
Loading test queries from data/ft/qa_pairs.jsonl
Evaluating dense retrieval...
Evaluating hybrid retrieval...
Results saved to: data/experiments/test/rag_quality_results.json
```

## Option 2: Run via CLI

The CLI provides more control and better progress output:

```bash
# Run RAG quality experiment
python run_experiments.py --experiments rag_quality --verbose

# View the results
python run_experiments.py --summary --output-dir data/experiments
```

## Option 3: Run All Available Experiments

**Warning**: This will take longer (10-30 minutes depending on dataset size)

```bash
python run_experiments.py --all --output-dir data/experiments/full_run
```

This runs:
- Dual-pipeline efficiency comparison
- RAG quality evaluation
- (More experiments as they're implemented)

## Understanding the Results

After running experiments, you'll find:

```
data/experiments/
├── all_results.json              # All results in JSON format
├── experiment_report.md          # Human-readable report
├── dual_pipeline/
│   ├── params.json              # Experiment parameters
│   ├── metrics.jsonl            # Time-series metrics
│   ├── summary.json             # Final summary
│   └── dual_pipeline_results.json  # Detailed results
└── rag_quality/
    ├── params.json
    ├── metrics.jsonl
    ├── summary.json
    └── rag_quality_results.json
```

### Reading Results

**Dual-Pipeline Results**:
```bash
python -c "
import json
with open('data/experiments/dual_pipeline/dual_pipeline_results.json') as f:
    results = json.load(f)

gains = results['efficiency_gains']
print(f'Time savings: {gains[\"time_saving_percent\"]:.2f}%')
print(f'GPU savings: {gains[\"gpu_saving_percent\"]:.2f}%')
print(f'Speedup factor: {gains[\"speedup_factor\"]:.2f}x')
"
```

**RAG Quality Results**:
```bash
python -c "
import json
with open('data/experiments/rag_quality/rag_quality_results.json') as f:
    results = json.load(f)

for variant, data in results['results'].items():
    metrics = data['metrics']
    recall_5 = metrics['recall_at_k'].get('recall@5', 0)
    mrr = metrics['mrr']
    print(f'{variant:20s} Recall@5: {recall_5:.4f}, MRR: {mrr:.4f}')
"
```

## Troubleshooting

### Issue: "No test queries found"

**Solution**: Generate QA pairs first by running the ETL pipeline:
```bash
cd liquid-etl-pipeline
python -m etl_pipeline.run_etl
# Or:
liquid-etl

# This creates data/ft/qa_pairs.jsonl which is used for evaluation
```

### Issue: "ModuleNotFoundError: No module named 'scipy'"

**Solution**: Reinstall dependencies:
```bash
cd liquid-shared-core
pip install -e .
# Or explicitly:
pip install scipy rouge-score numpy
```

### Issue: "Vector database not found"

**Solution**: Ensure ETL pipeline has been run at least once:
```bash
cd liquid-etl-pipeline
python -m etl_pipeline.run_etl
```

### Issue: "MLflow connection error"

**Solution**: This is OK! The framework falls back to local-only tracking. To use MLflow:
```bash
# Start MLflow server
mlflow ui --backend-store-uri file://./mlruns &

# Run experiments with MLflow
python run_experiments.py --all --mlflow-uri http://localhost:5000

# Open browser to http://localhost:5000 to view results
```

## Next Steps

### 1. Generate More Test Data

For better statistical power, generate more documents:
```bash
# Add more documents to data/raw/
cp /path/to/more/documents/* data/raw/

# Re-run ETL
cd liquid-etl-pipeline
python -m etl_pipeline.run_etl
```

### 2. Run Multiple Trials

For statistical significance, run experiments multiple times:
```python
from liquid_shared.experiments import ExperimentConfig, ModelConfig, DatasetConfig, EvalConfig
from liquid_shared.experiments.orchestrator import run_research_validation

config = ExperimentConfig(
    name="rag_quality_3_runs",
    description="RAG quality with 3 runs for significance testing",
    experiment_type="rag_quality",
    model=ModelConfig(name="LFM2-700M"),
    dataset=DatasetConfig(name="default"),
    eval=EvalConfig(recall_k_values=[1, 3, 5, 10]),
    n_runs=3,  # Run 3 times
    seed=42,
)

# Run with config
# (This requires modifying the orchestrator to accept custom configs)
```

### 3. Analyze Statistical Significance

```python
from liquid_shared.experiments import StatisticalTests, compare_multiple_methods

# Compare multiple retrieval methods
results = {
    "dense": [0.75, 0.76, 0.74, 0.75, 0.77],      # 5 runs
    "hybrid": [0.82, 0.83, 0.81, 0.84, 0.82],     # 5 runs
    "metadata": [0.86, 0.87, 0.85, 0.88, 0.86],   # 5 runs
}

comparison = compare_multiple_methods(results, alpha=0.05)
print(comparison)
```

### 4. Create Custom Experiments

See `liquid-shared-core/liquid_shared/experiments/README.md` for detailed examples of creating custom experiments.

### 5. Visualize Results with MLflow

```bash
# Start MLflow UI
mlflow ui --backend-store-uri file://./mlruns

# Open http://localhost:5000 in browser
# - Compare runs side-by-side
# - View parameter sweep results
# - Download artifacts and plots
```

## Example: Complete Workflow

Here's a complete workflow from data preparation to results:

```bash
# 1. Prepare data
echo "Step 1: Preparing data..."
# (Add documents to data/raw/)

# 2. Run ETL pipeline
echo "Step 2: Running ETL pipeline..."
cd liquid-etl-pipeline
python -m etl_pipeline.run_etl
cd ..

# 3. Run experiments
echo "Step 3: Running experiments..."
python run_experiments.py --all --output-dir data/experiments/$(date +%Y%m%d_%H%M%S)

# 4. View results
echo "Step 4: Results:"
python run_experiments.py --summary --output-dir data/experiments/$(ls -t data/experiments/ | head -1)

# 5. Start MLflow (optional)
echo "Step 5: Starting MLflow UI..."
mlflow ui --backend-store-uri file://./mlruns &
echo "Open http://localhost:5000 to view results"
```

## Getting Help

- **Framework docs**: `liquid-shared-core/liquid_shared/experiments/README.md`
- **Summary**: `RESEARCH_VALIDATION_SUMMARY.md`
- **CLI help**: `python run_experiments.py --help`
- **Verbose logging**: Add `--verbose` flag to any command

## What to Expect

### First Run (RAG Quality Only)
- **Duration**: 2-5 minutes (depends on dataset size)
- **Output**: Recall@K, MRR, NDCG metrics for different retrieval strategies
- **Files created**: ~5 JSON files with detailed results

### Full Run (All Experiments)
- **Duration**: 10-30 minutes
- **Output**: Dual-pipeline efficiency + RAG quality metrics
- **Files created**: ~10 JSON files + markdown report

### With Multiple Runs (n=5)
- **Duration**: 50-150 minutes
- **Output**: Statistical significance tests + confidence intervals
- **Files created**: ~50 JSON files + comprehensive report

## Success Criteria

After running experiments, you should have:
- ✓ JSON files with numerical results
- ✓ Markdown report with summary
- ✓ Metrics showing improvements (if any)
- ✓ Statistical tests (if multiple runs)
- ✓ All logs captured for reproducibility

You're ready to analyze results and write the paper! 🎉

## Pro Tips

1. **Start small**: Run with 10-20 documents first to verify everything works
2. **Use verbose mode**: Add `--verbose` to see detailed progress
3. **Check logs**: If something fails, check the log output for specific errors
4. **Save everything**: Each run creates a timestamped directory, never overwrites
5. **MLflow is optional**: Framework works fine without it, don't let connection errors block you

---

**Ready to run your first experiment?**

```bash
python run_experiments.py --experiments rag_quality --verbose
```

Good luck! 🚀
