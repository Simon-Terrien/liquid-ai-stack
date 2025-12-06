"""
MLflow experiment tracking integration for fine-tuning.

Tracks:
- Hyperparameters (learning rate, batch size, LoRA config)
- Training metrics (loss, eval loss, learning rate)
- Model artifacts
- System metrics (GPU memory, training time)
"""
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)

# Optional MLflow import
try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not available. Install with: uv add mlflow")


class MLflowTracker:
    """
    MLflow experiment tracking wrapper.

    Automatically tracks training metrics, hyperparameters, and models.
    """

    def __init__(
        self,
        experiment_name: str = "liquid-ft-training",
        tracking_uri: Optional[str] = None,
        run_name: Optional[str] = None,
    ):
        """
        Initialize MLflow tracker.

        Args:
            experiment_name: Name of MLflow experiment
            tracking_uri: MLflow tracking server URI (default: local file)
            run_name: Optional name for this run
        """
        self.enabled = MLFLOW_AVAILABLE
        self.run_name = run_name
        self.start_time = None

        if not self.enabled:
            logger.warning("MLflow tracking disabled (not installed)")
            return

        # Set tracking URI
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        # Set or create experiment
        try:
            mlflow.set_experiment(experiment_name)
            logger.info(f"MLflow experiment: {experiment_name}")
        except Exception as e:
            logger.error(f"Failed to set MLflow experiment: {e}")
            self.enabled = False

    def start_run(self, run_name: Optional[str] = None):
        """Start a new MLflow run."""
        if not self.enabled:
            return

        try:
            mlflow.start_run(run_name=run_name or self.run_name)
            self.start_time = time.time()
            logger.info(f"Started MLflow run: {mlflow.active_run().info.run_id}")
        except Exception as e:
            logger.error(f"Failed to start MLflow run: {e}")
            self.enabled = False

    def log_params(self, params: Dict[str, Any]):
        """
        Log hyperparameters.

        Args:
            params: Dictionary of parameter names and values
        """
        if not self.enabled:
            return

        try:
            # Flatten nested dicts and convert to strings
            flat_params = self._flatten_dict(params)
            mlflow.log_params(flat_params)
            logger.info(f"Logged {len(flat_params)} parameters to MLflow")
        except Exception as e:
            logger.error(f"Failed to log params: {e}")

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """
        Log a single metric.

        Args:
            key: Metric name
            value: Metric value
            step: Optional step/epoch number
        """
        if not self.enabled:
            return

        try:
            mlflow.log_metric(key, value, step=step)
        except Exception as e:
            logger.error(f"Failed to log metric {key}: {e}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log multiple metrics.

        Args:
            metrics: Dictionary of metric names and values
            step: Optional step/epoch number
        """
        if not self.enabled:
            return

        try:
            mlflow.log_metrics(metrics, step=step)
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")

    def log_system_metrics(self):
        """Log system metrics (GPU memory, etc.)."""
        if not self.enabled:
            return

        try:
            metrics = {}

            # GPU metrics
            if torch.cuda.is_available():
                metrics["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
                metrics["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / 1e9
                metrics["gpu_memory_max_allocated_gb"] = torch.cuda.max_memory_allocated() / 1e9

            # Training time
            if self.start_time:
                metrics["elapsed_time_minutes"] = (time.time() - self.start_time) / 60

            self.log_metrics(metrics)
        except Exception as e:
            logger.error(f"Failed to log system metrics: {e}")

    def log_model(self, model, artifact_path: str = "model"):
        """
        Log model artifact.

        Args:
            model: PyTorch model to log
            artifact_path: Path within MLflow run to store model
        """
        if not self.enabled:
            return

        try:
            mlflow.pytorch.log_model(model, artifact_path)
            logger.info(f"Logged model to MLflow: {artifact_path}")
        except Exception as e:
            logger.error(f"Failed to log model: {e}")

    def log_artifact(self, local_path: str | Path, artifact_path: Optional[str] = None):
        """
        Log a file artifact.

        Args:
            local_path: Path to local file
            artifact_path: Optional path within MLflow run
        """
        if not self.enabled:
            return

        try:
            mlflow.log_artifact(str(local_path), artifact_path)
            logger.info(f"Logged artifact: {local_path}")
        except Exception as e:
            logger.error(f"Failed to log artifact: {e}")

    def log_dict(self, dictionary: Dict, filename: str):
        """
        Log a dictionary as JSON artifact.

        Args:
            dictionary: Dictionary to log
            filename: Filename for the artifact
        """
        if not self.enabled:
            return

        try:
            mlflow.log_dict(dictionary, filename)
            logger.info(f"Logged dict as: {filename}")
        except Exception as e:
            logger.error(f"Failed to log dict: {e}")

    def end_run(self, status: str = "FINISHED"):
        """
        End the current MLflow run.

        Args:
            status: Run status (FINISHED, FAILED, KILLED)
        """
        if not self.enabled:
            return

        try:
            # Log final system metrics
            self.log_system_metrics()

            # End run
            mlflow.end_run(status=status)
            logger.info(f"Ended MLflow run with status: {status}")
        except Exception as e:
            logger.error(f"Failed to end run: {e}")

    def _flatten_dict(self, d: Dict, parent_key: str = "", sep: str = "_") -> Dict:
        """
        Flatten nested dictionary for MLflow logging.

        Args:
            d: Dictionary to flatten
            parent_key: Parent key for recursion
            sep: Separator for nested keys

        Returns:
            Flattened dictionary with string values
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k

            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                # Convert to string for MLflow
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


class MLflowCallback:
    """
    HuggingFace Trainer callback for MLflow logging.

    Automatically logs metrics during training.
    """

    def __init__(self, tracker: MLflowTracker):
        """
        Initialize callback.

        Args:
            tracker: MLflowTracker instance
        """
        self.tracker = tracker

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log metrics to MLflow."""
        if logs is None or not self.tracker.enabled:
            return

        # Extract step
        step = state.global_step

        # Log all metrics
        metrics_to_log = {}
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                metrics_to_log[key] = value

        if metrics_to_log:
            self.tracker.log_metrics(metrics_to_log, step=step)

    def on_epoch_end(self, args, state, control, **kwargs):
        """Log system metrics at end of epoch."""
        self.tracker.log_system_metrics()

    def on_train_end(self, args, state, control, **kwargs):
        """Log final metrics at end of training."""
        self.tracker.log_system_metrics()


def create_mlflow_tracker(
    experiment_name: str = "liquid-ft-training",
    run_name: Optional[str] = None,
    tracking_uri: Optional[str] = None,
) -> MLflowTracker:
    """
    Create MLflow tracker with sensible defaults.

    Args:
        experiment_name: Name of experiment
        run_name: Optional run name
        tracking_uri: Optional tracking server URI

    Returns:
        Configured MLflowTracker instance
    """
    return MLflowTracker(
        experiment_name=experiment_name,
        run_name=run_name,
        tracking_uri=tracking_uri,
    )
