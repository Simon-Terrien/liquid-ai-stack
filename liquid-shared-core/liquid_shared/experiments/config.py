"""Experiment configuration management.

Centralized configuration for all research experiments with type safety
and validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from liquid_shared import DATA_DIR


@dataclass
class ModelConfig:
    """Configuration for model selection in experiments."""

    name: str  # Model identifier (e.g., "LFM2-700M")
    device: str = "auto"  # "auto", "cuda", "cpu"
    dtype: str = "auto"  # "auto", "fp16", "bf16", "fp32"
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "name": self.name,
            "device": self.device,
            "dtype": self.dtype,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }


@dataclass
class DatasetConfig:
    """Configuration for dataset processing."""

    name: str
    data_dir: Path = DATA_DIR
    split: Literal["train", "val", "test", "all"] = "all"
    max_samples: int | None = None  # None = all samples
    shuffle: bool = False
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "name": self.name,
            "data_dir": str(self.data_dir),
            "split": self.split,
            "max_samples": self.max_samples,
            "shuffle": self.shuffle,
            "seed": self.seed,
        }


@dataclass
class EvalConfig:
    """Configuration for evaluation metrics."""

    # RAG metrics
    compute_recall_at_k: bool = True
    recall_k_values: list[int] = field(default_factory=lambda: [1, 3, 5, 10])
    compute_mrr: bool = True
    compute_ndcg: bool = True

    # Generation metrics
    compute_rouge: bool = True
    compute_bleu: bool = True
    compute_perplexity: bool = False  # Requires reference model

    # Statistical testing
    n_bootstrap_samples: int = 1000
    confidence_level: float = 0.95

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "compute_recall_at_k": self.compute_recall_at_k,
            "recall_k_values": self.recall_k_values,
            "compute_mrr": self.compute_mrr,
            "compute_ndcg": self.compute_ndcg,
            "compute_rouge": self.compute_rouge,
            "compute_bleu": self.compute_bleu,
            "compute_perplexity": self.compute_perplexity,
            "n_bootstrap_samples": self.n_bootstrap_samples,
            "confidence_level": self.confidence_level,
        }


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    name: str
    description: str
    experiment_type: Literal[
        "dual_pipeline",
        "multi_model_ablation",
        "rag_quality",
        "ft_quality",
        "chunking_ablation",
        "metadata_ablation",
        "hybrid_retrieval",
    ]

    # Component configs
    model: ModelConfig
    dataset: DatasetConfig
    eval: EvalConfig

    # Experiment settings
    output_dir: Path = DATA_DIR / "experiments"
    mlflow_tracking_uri: str | None = None
    mlflow_experiment_name: str | None = None
    n_runs: int = 1  # Number of runs for statistical significance
    seed: int = 42

    # Ablation-specific settings
    ablation_params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "name": self.name,
            "description": self.description,
            "experiment_type": self.experiment_type,
            "model": self.model.to_dict(),
            "dataset": self.dataset.to_dict(),
            "eval": self.eval.to_dict(),
            "output_dir": str(self.output_dir),
            "mlflow_tracking_uri": self.mlflow_tracking_uri,
            "mlflow_experiment_name": self.mlflow_experiment_name,
            "n_runs": self.n_runs,
            "seed": self.seed,
            "ablation_params": self.ablation_params,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        """Create config from dictionary."""
        return cls(
            name=data["name"],
            description=data["description"],
            experiment_type=data["experiment_type"],
            model=ModelConfig(**data["model"]),
            dataset=DatasetConfig(**data["dataset"]),
            eval=EvalConfig(**data["eval"]),
            output_dir=Path(data.get("output_dir", DATA_DIR / "experiments")),
            mlflow_tracking_uri=data.get("mlflow_tracking_uri"),
            mlflow_experiment_name=data.get("mlflow_experiment_name"),
            n_runs=data.get("n_runs", 1),
            seed=data.get("seed", 42),
            ablation_params=data.get("ablation_params", {}),
        )
