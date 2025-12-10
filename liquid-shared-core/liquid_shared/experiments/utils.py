"""Utility functions for experiments."""

from __future__ import annotations

import logging
import os
import platform
import random
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Make CUDA operations deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info(f"Set random seed to {seed}")


def get_gpu_info() -> dict[str, Any]:
    """
    Get GPU information for logging.

    Returns:
        Dictionary with GPU information
    """
    if not torch.cuda.is_available():
        return {"available": False}

    info = {
        "available": True,
        "count": torch.cuda.device_count(),
        "devices": [],
    }

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        device_info = {
            "index": i,
            "name": props.name,
            "total_memory_gb": props.total_memory / (1024**3),
            "compute_capability": f"{props.major}.{props.minor}",
        }
        info["devices"].append(device_info)

    return info


def log_system_info() -> dict[str, Any]:
    """
    Log system information for reproducibility.

    Returns:
        Dictionary with system information
    """
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_info": get_gpu_info(),
        "cpu_count": os.cpu_count(),
    }

    logger.info(f"System info: {info}")
    return info


def format_metric_table(
    results: dict[str, dict[str, float]],
    metric_names: list[str] | None = None,
) -> str:
    """
    Format experiment results as a table.

    Args:
        results: Dictionary mapping method names to metric dictionaries
        metric_names: List of metrics to include (None = all)

    Returns:
        Formatted table string
    """
    if not results:
        return "No results to display"

    # Collect all metrics if not specified
    if metric_names is None:
        all_metrics = set()
        for metrics in results.values():
            all_metrics.update(metrics.keys())
        metric_names = sorted(all_metrics)

    # Build table
    lines = []

    # Header
    header = ["Method"] + metric_names
    lines.append(" | ".join(header))
    lines.append("-" * (sum(len(h) for h in header) + 3 * len(header)))

    # Rows
    for method_name, metrics in results.items():
        row = [method_name]
        for metric_name in metric_names:
            value = metrics.get(metric_name, 0.0)
            row.append(f"{value:.4f}")
        lines.append(" | ".join(row))

    return "\n".join(lines)


def save_results_markdown(
    results: dict[str, Any],
    output_path: str,
    title: str = "Experiment Results",
) -> None:
    """
    Save experiment results as markdown report.

    Args:
        results: Results dictionary
        output_path: Path to save markdown file
        title: Report title
    """
    from pathlib import Path

    lines = [
        f"# {title}",
        "",
        "## Configuration",
        "",
        "```json",
        str(results.get("config", {})),
        "```",
        "",
        "## Results",
        "",
    ]

    # Add metrics table if available
    if "metrics" in results:
        lines.append("### Metrics")
        lines.append("")
        lines.append(format_metric_table(results["metrics"]))
        lines.append("")

    # Add statistical tests if available
    if "statistical_tests" in results:
        lines.append("### Statistical Significance")
        lines.append("")
        for test_name, test_result in results["statistical_tests"].items():
            lines.append(f"**{test_name}**")
            lines.append(f"- p-value: {test_result.get('p_value', 'N/A')}")
            lines.append(f"- Significant: {test_result.get('is_significant', False)}")
            if "effect_size" in test_result:
                lines.append(f"- Effect size: {test_result['effect_size']}")
            lines.append("")

    Path(output_path).write_text("\n".join(lines))
    logger.info(f"Saved results to {output_path}")
