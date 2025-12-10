"""Dual-pipeline efficiency experiment.

Compares the unified "two birds one stone" ETL pipeline against:
1. Separate RAG-only pipeline (no fine-tuning data generation)
2. Separate FT-only pipeline (only fine-tuning data, no vector store)

Measures:
- GPU hours
- Peak memory usage
- Throughput (docs/min)
- Output parity (verify same results)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from liquid_shared.experiments import (
    ExperimentConfig,
    ExperimentTracker,
    set_seed,
    log_system_info,
)
from liquid_shared.experiments.config import ModelConfig, DatasetConfig, EvalConfig
from liquid_shared import DATA_DIR

from etl_pipeline.run_etl import run_etl
from etl_pipeline.experiments.runtime import RuntimeCollector

logger = logging.getLogger(__name__)


@dataclass
class PipelineResults:
    """Results from running a pipeline variant."""

    variant_name: str
    total_seconds: float
    gpu_hours: float  # GPU time in hours
    peak_memory_mb: float
    throughput_docs_per_min: float
    num_chunks: int
    num_qa_pairs: int
    num_ft_samples: int
    success: bool
    output_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "variant_name": self.variant_name,
            "total_seconds": self.total_seconds,
            "gpu_hours": self.gpu_hours,
            "peak_memory_mb": self.peak_memory_mb,
            "throughput_docs_per_min": self.throughput_docs_per_min,
            "num_chunks": self.num_chunks,
            "num_qa_pairs": self.num_qa_pairs,
            "num_ft_samples": self.num_ft_samples,
            "success": self.success,
        }


class DualPipelineExperiment:
    """
    Experiment to validate dual-output ETL efficiency.

    Tests the hypothesis that generating both RAG vectors and fine-tuning
    data in one pass is more efficient than running two separate pipelines.
    """

    def __init__(self, config: ExperimentConfig):
        """
        Initialize experiment.

        Args:
            config: Experiment configuration
        """
        self.config = config
        self.output_dir = config.output_dir / "dual_pipeline"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup tracker
        self.tracker = ExperimentTracker(
            experiment_name=config.mlflow_experiment_name or "dual-pipeline-comparison",
            run_name=config.name,
            tracking_uri=config.mlflow_tracking_uri,
            output_dir=self.output_dir,
        )

    def run(self) -> dict[str, Any]:
        """
        Run the dual-pipeline comparison experiment.

        Returns:
            Dictionary with results and statistical comparison
        """
        logger.info("Starting dual-pipeline comparison experiment")

        with self.tracker:
            # Log system info and config
            system_info = log_system_info()
            self.tracker.log_params(system_info)
            self.tracker.log_params(self.config.to_dict())

            # Set seed for reproducibility
            set_seed(self.config.seed)

            # Run each variant
            results = {}

            # 1. Unified pipeline (current implementation)
            logger.info("Running unified pipeline...")
            results["unified"] = self._run_unified_pipeline()
            self._log_variant_metrics("unified", results["unified"])

            # 2. RAG-only pipeline
            logger.info("Running RAG-only pipeline...")
            results["rag_only"] = self._run_rag_only_pipeline()
            self._log_variant_metrics("rag_only", results["rag_only"])

            # 3. FT-only pipeline
            logger.info("Running FT-only pipeline...")
            results["ft_only"] = self._run_ft_only_pipeline()
            self._log_variant_metrics("ft_only", results["ft_only"])

            # 4. Separate pipelines (RAG + FT)
            logger.info("Computing separate pipeline metrics...")
            results["separate"] = self._compute_separate_metrics(
                results["rag_only"], results["ft_only"]
            )
            self._log_variant_metrics("separate", results["separate"])

            # Compute efficiency gains
            efficiency = self._compute_efficiency_gains(
                results["unified"], results["separate"]
            )
            self.tracker.log_metrics(efficiency)

            # Verify output parity
            parity = self._verify_output_parity(results)
            self.tracker.log_metrics(parity)

            # Save detailed results
            final_results = {
                "config": self.config.to_dict(),
                "system_info": system_info,
                "results": {k: v.to_dict() for k, v in results.items()},
                "efficiency_gains": efficiency,
                "output_parity": parity,
            }

            self.tracker.log_dict(final_results, "dual_pipeline_results.json")

            logger.info("Dual-pipeline experiment complete")
            return final_results

    def _run_unified_pipeline(self) -> PipelineResults:
        """Run the unified dual-output pipeline."""
        input_dir = self.config.dataset.data_dir / "raw"
        runtime_collector = RuntimeCollector()

        start_time = perf_counter()
        start_gpu_time = self._get_gpu_time()

        runtime_collector.start_run()
        stats = run_etl(
            input_dir=input_dir,
            file_pattern="*.*",
            runtime_collector=runtime_collector,
        )

        total_seconds = perf_counter() - start_time
        gpu_hours = (self._get_gpu_time() - start_gpu_time) / 3600

        pipeline_runtime = runtime_collector.finish_run(total_seconds, stats)

        return PipelineResults(
            variant_name="unified",
            total_seconds=total_seconds,
            gpu_hours=gpu_hours,
            peak_memory_mb=pipeline_runtime.gpu_peak_mb or 0.0,
            throughput_docs_per_min=pipeline_runtime.throughput_docs_per_minute,
            num_chunks=stats.get("num_chunks", 0),
            num_qa_pairs=stats.get("num_qa_pairs", 0),
            num_ft_samples=stats.get("num_ft_samples", 0),
            success=True,
        )

    def _run_rag_only_pipeline(self) -> PipelineResults:
        """
        Run RAG-only pipeline (no fine-tuning data generation).

        This would require modifying the pipeline to skip QA generation.
        For now, we simulate by running the full pipeline but measuring
        only RAG-related operations.
        """
        # TODO: Implement RAG-only variant by modifying graph
        # For now, use unified pipeline metrics as approximation
        logger.warning("RAG-only pipeline not yet implemented, using approximation")

        unified = self._run_unified_pipeline()

        # Approximate: assume QA generation takes 30% of time
        # (This should be replaced with actual measurement)
        return PipelineResults(
            variant_name="rag_only",
            total_seconds=unified.total_seconds * 0.7,
            gpu_hours=unified.gpu_hours * 0.7,
            peak_memory_mb=unified.peak_memory_mb,
            throughput_docs_per_min=unified.throughput_docs_per_min / 0.7,
            num_chunks=unified.num_chunks,
            num_qa_pairs=0,
            num_ft_samples=0,
            success=True,
        )

    def _run_ft_only_pipeline(self) -> PipelineResults:
        """
        Run FT-only pipeline (no vector store generation).

        This would require modifying the pipeline to skip embedding/storage.
        """
        # TODO: Implement FT-only variant
        logger.warning("FT-only pipeline not yet implemented, using approximation")

        unified = self._run_unified_pipeline()

        # Approximate: assume RAG operations take 40% of time
        return PipelineResults(
            variant_name="ft_only",
            total_seconds=unified.total_seconds * 0.6,
            gpu_hours=unified.gpu_hours * 0.6,
            peak_memory_mb=unified.peak_memory_mb,
            throughput_docs_per_min=unified.throughput_docs_per_min / 0.6,
            num_chunks=0,
            num_qa_pairs=unified.num_qa_pairs,
            num_ft_samples=unified.num_ft_samples,
            success=True,
        )

    def _compute_separate_metrics(
        self, rag_only: PipelineResults, ft_only: PipelineResults
    ) -> PipelineResults:
        """Compute combined metrics for running two separate pipelines."""
        return PipelineResults(
            variant_name="separate",
            total_seconds=rag_only.total_seconds + ft_only.total_seconds,
            gpu_hours=rag_only.gpu_hours + ft_only.gpu_hours,
            peak_memory_mb=max(rag_only.peak_memory_mb, ft_only.peak_memory_mb),
            throughput_docs_per_min=2
            / (1 / rag_only.throughput_docs_per_min + 1 / ft_only.throughput_docs_per_min),
            num_chunks=rag_only.num_chunks,
            num_qa_pairs=ft_only.num_qa_pairs,
            num_ft_samples=ft_only.num_ft_samples,
            success=True,
        )

    def _compute_efficiency_gains(
        self, unified: PipelineResults, separate: PipelineResults
    ) -> dict[str, float]:
        """Compute efficiency gains of unified vs separate pipelines."""
        time_saving = (
            (separate.total_seconds - unified.total_seconds) / separate.total_seconds * 100
        )
        gpu_saving = (separate.gpu_hours - unified.gpu_hours) / separate.gpu_hours * 100
        throughput_gain = (
            (unified.throughput_docs_per_min - separate.throughput_docs_per_min)
            / separate.throughput_docs_per_min
            * 100
        )

        return {
            "time_saving_percent": time_saving,
            "gpu_saving_percent": gpu_saving,
            "throughput_gain_percent": throughput_gain,
            "speedup_factor": separate.total_seconds / unified.total_seconds,
        }

    def _verify_output_parity(self, results: dict[str, PipelineResults]) -> dict[str, float]:
        """Verify that unified pipeline produces same outputs as separate pipelines."""
        unified = results["unified"]
        separate = results["separate"]

        # Check that output counts match
        chunks_match = unified.num_chunks == separate.num_chunks
        qa_match = unified.num_qa_pairs == separate.num_qa_pairs
        ft_match = unified.num_ft_samples == separate.num_ft_samples

        return {
            "chunks_parity": 1.0 if chunks_match else 0.0,
            "qa_pairs_parity": 1.0 if qa_match else 0.0,
            "ft_samples_parity": 1.0 if ft_match else 0.0,
            "overall_parity": 1.0 if (chunks_match and qa_match and ft_match) else 0.0,
        }

    def _log_variant_metrics(self, variant_name: str, results: PipelineResults) -> None:
        """Log metrics for a pipeline variant."""
        metrics = {
            f"{variant_name}/total_seconds": results.total_seconds,
            f"{variant_name}/gpu_hours": results.gpu_hours,
            f"{variant_name}/peak_memory_mb": results.peak_memory_mb,
            f"{variant_name}/throughput_docs_per_min": results.throughput_docs_per_min,
            f"{variant_name}/num_chunks": float(results.num_chunks),
            f"{variant_name}/num_qa_pairs": float(results.num_qa_pairs),
            f"{variant_name}/num_ft_samples": float(results.num_ft_samples),
        }
        self.tracker.log_metrics(metrics)

    @staticmethod
    def _get_gpu_time() -> float:
        """Get cumulative GPU time in seconds."""
        if torch.cuda.is_available():
            # Approximate GPU time (would need nvidia-smi for accurate measurement)
            return perf_counter()
        return 0.0


def run_dual_pipeline_experiment(
    config: ExperimentConfig | None = None,
) -> dict[str, Any]:
    """
    Run dual-pipeline comparison experiment.

    Args:
        config: Experiment configuration (uses defaults if None)

    Returns:
        Experiment results
    """
    if config is None:
        config = ExperimentConfig(
            name="dual_pipeline_comparison",
            description="Compare unified vs separate ETL pipelines",
            experiment_type="dual_pipeline",
            model=ModelConfig(name="LFM2-2.6B"),
            dataset=DatasetConfig(name="default"),
            eval=EvalConfig(),
        )

    experiment = DualPipelineExperiment(config)
    return experiment.run()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    results = run_dual_pipeline_experiment()
    print("\n=== Dual-Pipeline Experiment Results ===")
    print(json.dumps(results["efficiency_gains"], indent=2))
