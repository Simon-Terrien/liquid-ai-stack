"""Configuration for metadata ablation study."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AblationConfig:
    """Configuration for ablation study experiments."""

    # Data paths
    queries_path: Path = Path("experiments/data/test_queries.json")
    judgments_path: Path = Path("experiments/data/relevance_judgments.json")
    results_path: Path = Path("experiments/data/results")

    # Vector store config
    vectordb_path: Path = Path("liquid-shared-core/data/vectordb")
    collection_name: str = "documents"

    # Retrieval config
    top_k: int = 10
    vector_weight: float = 0.7
    bm25_weight: float = 0.3
    importance_weight: float = 0.2

    # Relevance judgment config
    llm_model: str = "LiquidAI/LFM2-1.2B"
    relevance_threshold: float = 0.5
    use_hybrid_judgments: bool = True

    # Evaluation metrics
    k_values: list[int] = field(default_factory=lambda: [1, 3, 5, 10])
    ndcg_k_values: list[int] = field(default_factory=lambda: [5, 10])

    # Statistical testing
    significance_level: float = 0.05
    confidence_level: float = 0.95

    # Variants to test
    variants: list[str] = field(
        default_factory=lambda: ["v0", "v1", "v2", "v3", "v4", "v5"]
    )

    def __post_init__(self):
        """Ensure paths are Path objects."""
        self.queries_path = Path(self.queries_path)
        self.judgments_path = Path(self.judgments_path)
        self.results_path = Path(self.results_path)
        self.vectordb_path = Path(self.vectordb_path)

        # Create results directory if it doesn't exist
        self.results_path.mkdir(parents=True, exist_ok=True)
