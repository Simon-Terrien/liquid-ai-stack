"""Master experiment orchestrator for running all research validation experiments."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .utils import save_results_markdown

logger = logging.getLogger(__name__)


class ExperimentOrchestrator:
    """
    Orchestrates multiple experiments for comprehensive research validation.

    Manages:
    - Dual-pipeline efficiency comparison
    - Multi-model ablation studies
    - RAG quality evaluation
    - Fine-tuning data quality assessment
    - Statistical analysis and reporting
    """

    def __init__(
        self,
        output_dir: Path,
        mlflow_tracking_uri: str | None = None,
    ):
        """
        Initialize orchestrator.

        Args:
            output_dir: Root directory for all experiment outputs
            mlflow_tracking_uri: Optional MLflow tracking server URI
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mlflow_tracking_uri = mlflow_tracking_uri

        self.results: dict[str, Any] = {}

    def run_all_experiments(
        self,
        experiments: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Run all configured experiments.

        Args:
            experiments: List of experiment names to run (None = all)
                       Options: "dual_pipeline", "rag_quality", "multi_model", "ft_quality"

        Returns:
            Aggregated results from all experiments
        """
        if experiments is None:
            experiments = ["dual_pipeline", "rag_quality"]  # Start with these

        logger.info(f"Starting experiment orchestration: {experiments}")

        for exp_name in experiments:
            try:
                logger.info(f"Running experiment: {exp_name}")
                result = self._run_experiment(exp_name)
                self.results[exp_name] = result
                logger.info(f"Completed experiment: {exp_name}")
            except Exception as e:
                logger.error(f"Failed to run experiment {exp_name}: {e}", exc_info=True)
                self.results[exp_name] = {"error": str(e)}

        # Generate aggregated report
        self._generate_report()

        logger.info("All experiments complete")
        return self.results

    def _run_experiment(self, exp_name: str) -> dict[str, Any]:
        """Run a single experiment by name."""
        if exp_name == "dual_pipeline":
            return self._run_dual_pipeline()
        elif exp_name == "rag_quality":
            return self._run_rag_quality()
        elif exp_name == "multi_model":
            return self._run_multi_model_ablation()
        elif exp_name == "ft_quality":
            return self._run_ft_quality()
        else:
            raise ValueError(f"Unknown experiment: {exp_name}")

    def _run_dual_pipeline(self) -> dict[str, Any]:
        """Run dual-pipeline efficiency experiment."""
        try:
            from etl_pipeline.experiments.dual_pipeline_experiment import (
                run_dual_pipeline_experiment,
                ExperimentConfig,
                ModelConfig,
                DatasetConfig,
                EvalConfig,
            )

            config = ExperimentConfig(
                name="dual_pipeline_efficiency",
                description="Compare unified vs separate ETL pipelines",
                experiment_type="dual_pipeline",
                model=ModelConfig(name="LFM2-2.6B"),
                dataset=DatasetConfig(name="default"),
                eval=EvalConfig(),
                output_dir=self.output_dir,
                mlflow_tracking_uri=self.mlflow_tracking_uri,
            )

            return run_dual_pipeline_experiment(config)
        except ImportError as e:
            logger.error(f"Failed to import dual_pipeline_experiment: {e}")
            return {"error": str(e)}

    def _run_rag_quality(self) -> dict[str, Any]:
        """Run RAG quality evaluation experiment."""
        try:
            from rag_runtime.experiments.rag_quality_experiment import (
                run_rag_quality_experiment,
                ExperimentConfig,
                ModelConfig,
                DatasetConfig,
                EvalConfig,
            )

            config = ExperimentConfig(
                name="rag_quality_evaluation",
                description="Evaluate RAG retrieval quality",
                experiment_type="rag_quality",
                model=ModelConfig(name="LFM2-700M"),
                dataset=DatasetConfig(name="default"),
                eval=EvalConfig(recall_k_values=[1, 3, 5, 10, 20]),
                output_dir=self.output_dir,
                mlflow_tracking_uri=self.mlflow_tracking_uri,
            )

            return run_rag_quality_experiment(config)
        except ImportError as e:
            logger.error(f"Failed to import rag_quality_experiment: {e}")
            return {"error": str(e)}

    def _run_multi_model_ablation(self) -> dict[str, Any]:
        """Run multi-model ablation study."""
        logger.warning("Multi-model ablation not yet implemented")
        return {"status": "not_implemented"}

    def _run_ft_quality(self) -> dict[str, Any]:
        """Run fine-tuning data quality assessment."""
        logger.warning("FT quality assessment not yet implemented")
        return {"status": "not_implemented"}

    def _generate_report(self) -> None:
        """Generate aggregated markdown report."""
        report_path = self.output_dir / "experiment_report.md"

        logger.info(f"Generating report: {report_path}")
        save_results_markdown(
            self.results,
            str(report_path),
            title="Research Validation Experiment Results",
        )

        # Also save JSON
        json_path = self.output_dir / "all_results.json"
        json_path.write_text(json.dumps(self.results, indent=2, default=str))

        logger.info(f"Report saved to {report_path}")


def run_research_validation(
    output_dir: str | Path = "data/experiments",
    experiments: list[str] | None = None,
    mlflow_tracking_uri: str | None = None,
) -> dict[str, Any]:
    """
    Run all research validation experiments.

    This is the main entry point for executing the comprehensive
    experimental validation required for publication.

    Args:
        output_dir: Directory for experiment outputs
        experiments: List of experiments to run (None = all)
        mlflow_tracking_uri: Optional MLflow tracking server

    Returns:
        Aggregated results from all experiments
    """
    orchestrator = ExperimentOrchestrator(
        output_dir=Path(output_dir),
        mlflow_tracking_uri=mlflow_tracking_uri,
    )

    return orchestrator.run_all_experiments(experiments=experiments)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    results = run_research_validation()
    print("\n=== Research Validation Complete ===")
    print(f"Results saved to: data/experiments/")
