"""Runtime measurement utilities for ETL research experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import util
from pathlib import Path
from typing import Any

if util.find_spec("torch"):
    import torch
else:  # pragma: no cover - optional dependency guard
    torch = None


@dataclass
class DocumentRuntime:
    """Runtime and quality metrics for a single document run."""

    path: str
    wall_seconds: float
    num_chunks: int
    num_qa_pairs: int
    num_ft_samples: int
    qa_pass_rate: float | None
    success: bool


@dataclass
class PipelineRuntime:
    """Aggregated runtime metrics across documents."""

    total_seconds: float
    throughput_docs_per_minute: float
    documents: list[DocumentRuntime] = field(default_factory=list)
    aggregate_stats: dict[str, Any] = field(default_factory=dict)
    gpu_peak_mb: float | None = None


class RuntimeCollector:
    """Collects per-document and overall runtime metrics."""

    def __init__(self) -> None:
        self.documents: list[DocumentRuntime] = []
        self._gpu_available = bool(torch and torch.cuda.is_available())

    def start_run(self) -> None:
        """Reset GPU peak tracking when available."""
        if self._gpu_available:
            torch.cuda.reset_peak_memory_stats()  # type: ignore[call-arg]

    def record_document(self, path: Path, result: Any, wall_seconds: float) -> None:
        """Record runtime statistics for a processed document."""
        metrics = DocumentRuntime(
            path=str(path),
            wall_seconds=wall_seconds,
            num_chunks=len(result.chunks) if result else 0,
            num_qa_pairs=len(result.qa_pairs) if result else 0,
            num_ft_samples=len(result.ft_samples) if result else 0,
            qa_pass_rate=(result.stats.get("qa_pass_rate") if result else None),
            success=result is not None,
        )
        self.documents.append(metrics)

    def finish_run(self, total_seconds: float, aggregate_stats: dict[str, Any]) -> PipelineRuntime:
        """Summarize collected runtime data."""
        throughput = 0.0
        if total_seconds > 0 and self.documents:
            throughput = len(self.documents) / (total_seconds / 60)

        gpu_peak = None
        if self._gpu_available:
            gpu_peak = torch.cuda.max_memory_allocated() / (1024**2)

        return PipelineRuntime(
            total_seconds=total_seconds,
            throughput_docs_per_minute=throughput,
            documents=self.documents.copy(),
            aggregate_stats=aggregate_stats,
            gpu_peak_mb=gpu_peak,
        )


def pipeline_runtime_to_dict(runtime: PipelineRuntime) -> dict[str, Any]:
    """Convert a PipelineRuntime into a JSON-serializable dict."""
    return {
        "total_seconds": runtime.total_seconds,
        "throughput_docs_per_minute": runtime.throughput_docs_per_minute,
        "gpu_peak_mb": runtime.gpu_peak_mb,
        "aggregate_stats": runtime.aggregate_stats,
        "documents": [
            {
                "path": doc.path,
                "wall_seconds": doc.wall_seconds,
                "num_chunks": doc.num_chunks,
                "num_qa_pairs": doc.num_qa_pairs,
                "num_ft_samples": doc.num_ft_samples,
                "qa_pass_rate": doc.qa_pass_rate,
                "success": doc.success,
            }
            for doc in runtime.documents
        ],
    }
