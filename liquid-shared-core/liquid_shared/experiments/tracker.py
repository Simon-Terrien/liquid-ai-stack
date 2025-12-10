"""Experiment tracking with MLflow integration.

Extends the fine-tuning MLflow tracker to support all experiment types.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Optional MLflow import
try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not available. Install with: uv add mlflow")


class ExperimentTracker:
    """
    Unified experiment tracker for all research experiments.

    Tracks metrics, parameters, and artifacts to both MLflow and local files
    for reproducibility and publication.
    """

    def __init__(
        self,
        experiment_name: str,
        run_name: str | None = None,
        tracking_uri: str | None = None,
        output_dir: Path | None = None,
    ):
        """
        Initialize experiment tracker.

        Args:
            experiment_name: Name of MLflow experiment
            run_name: Optional name for this run
            tracking_uri: MLflow tracking server URI
            output_dir: Local directory for saving results
        """
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.output_dir = output_dir
        self.start_time = time.time()
        self.metrics_buffer: dict[str, list[tuple[float, int | None]]] = {}
        self.params_buffer: dict[str, Any] = {}

        # MLflow setup
        self.mlflow_enabled = MLFLOW_AVAILABLE
        if self.mlflow_enabled:
            try:
                if tracking_uri:
                    mlflow.set_tracking_uri(tracking_uri)
                mlflow.set_experiment(experiment_name)
            except Exception as e:
                logger.error(f"Failed to setup MLflow: {e}")
                self.mlflow_enabled = False

        # Local output setup
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            self.metrics_file = output_dir / "metrics.jsonl"
            self.params_file = output_dir / "params.json"
            self.summary_file = output_dir / "summary.json"

    def start_run(self, run_name: str | None = None) -> None:
        """Start a new tracking run."""
        if self.mlflow_enabled:
            try:
                mlflow.start_run(run_name=run_name or self.run_name)
                logger.info(f"Started MLflow run: {mlflow.active_run().info.run_id}")
            except Exception as e:
                logger.error(f"Failed to start MLflow run: {e}")
                self.mlflow_enabled = False

        self.start_time = time.time()

    def log_params(self, params: dict[str, Any]) -> None:
        """Log experiment parameters."""
        self.params_buffer.update(params)

        if self.mlflow_enabled:
            try:
                flat_params = self._flatten_dict(params)
                mlflow.log_params(flat_params)
            except Exception as e:
                logger.error(f"Failed to log params to MLflow: {e}")

        # Save locally
        if self.output_dir:
            self.params_file.write_text(
                json.dumps(self.params_buffer, indent=2, default=str)
            )

    def log_metric(
        self, key: str, value: float, step: int | None = None
    ) -> None:
        """Log a single metric."""
        # Buffer for local saving
        if key not in self.metrics_buffer:
            self.metrics_buffer[key] = []
        self.metrics_buffer[key].append((value, step))

        # Log to MLflow
        if self.mlflow_enabled:
            try:
                mlflow.log_metric(key, value, step=step)
            except Exception as e:
                logger.error(f"Failed to log metric {key} to MLflow: {e}")

    def log_metrics(
        self, metrics: dict[str, float], step: int | None = None
    ) -> None:
        """Log multiple metrics."""
        for key, value in metrics.items():
            self.log_metric(key, value, step=step)

    def log_artifact(self, file_path: Path | str, artifact_path: str | None = None) -> None:
        """Log a file artifact."""
        if self.mlflow_enabled:
            try:
                mlflow.log_artifact(str(file_path), artifact_path)
            except Exception as e:
                logger.error(f"Failed to log artifact to MLflow: {e}")

    def log_dict(self, data: dict, filename: str) -> None:
        """Log a dictionary as JSON artifact."""
        if self.mlflow_enabled:
            try:
                mlflow.log_dict(data, filename)
            except Exception as e:
                logger.error(f"Failed to log dict to MLflow: {e}")

        # Save locally
        if self.output_dir:
            output_path = self.output_dir / filename
            output_path.write_text(json.dumps(data, indent=2, default=str))

    def end_run(self, status: str = "FINISHED") -> dict[str, Any]:
        """
        End the tracking run and save summary.

        Returns:
            Summary dictionary with all tracked metrics and parameters
        """
        duration = time.time() - self.start_time

        # Create summary
        summary = {
            "experiment_name": self.experiment_name,
            "run_name": self.run_name,
            "status": status,
            "duration_seconds": duration,
            "params": self.params_buffer,
            "metrics_summary": self._compute_metrics_summary(),
        }

        # Save summary locally
        if self.output_dir:
            self.summary_file.write_text(json.dumps(summary, indent=2, default=str))
            # Save detailed metrics
            self._save_metrics_to_file()

        # End MLflow run
        if self.mlflow_enabled:
            try:
                mlflow.log_metrics({"duration_seconds": duration})
                mlflow.end_run(status=status)
                logger.info(f"Ended MLflow run with status: {status}")
            except Exception as e:
                logger.error(f"Failed to end MLflow run: {e}")

        return summary

    def _compute_metrics_summary(self) -> dict[str, Any]:
        """Compute summary statistics for all tracked metrics."""
        summary = {}
        for key, values in self.metrics_buffer.items():
            vals = [v[0] for v in values]
            if vals:
                summary[key] = {
                    "count": len(vals),
                    "mean": sum(vals) / len(vals),
                    "min": min(vals),
                    "max": max(vals),
                    "final": vals[-1],
                }
        return summary

    def _save_metrics_to_file(self) -> None:
        """Save all metrics to JSONL file."""
        if not self.output_dir:
            return

        with self.metrics_file.open("w") as f:
            for key, values in self.metrics_buffer.items():
                for value, step in values:
                    record = {
                        "metric": key,
                        "value": value,
                        "step": step,
                        "timestamp": time.time(),
                    }
                    f.write(json.dumps(record) + "\n")

    def _flatten_dict(
        self, d: dict, parent_key: str = "", sep: str = "_"
    ) -> dict[str, str]:
        """Flatten nested dictionary for MLflow."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, str(v)))
        return dict(items)

    def __enter__(self):
        """Context manager entry."""
        self.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type is not None:
            self.end_run(status="FAILED")
        else:
            self.end_run(status="FINISHED")
