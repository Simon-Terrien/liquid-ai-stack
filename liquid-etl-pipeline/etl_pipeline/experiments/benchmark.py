"""Benchmark runner for research experiments.

Measures wall-clock throughput of the unified ETL pipeline and optionally
persists metrics for later analysis.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from time import perf_counter

from etl_pipeline.run_etl import RAW_DIR, run_etl

from .runtime import RuntimeCollector, pipeline_runtime_to_dict

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = RAW_DIR.parent / "processed" / "etl_benchmark.json"


def run_benchmark(input_dir: Path, pattern: str, output_path: Path | None) -> None:
    """Execute the ETL pipeline with runtime instrumentation."""
    runtime_collector = RuntimeCollector()
    start_time = perf_counter()
    stats = run_etl(
        input_dir=input_dir,
        file_pattern=pattern,
        runtime_collector=runtime_collector,
    )
    total_seconds = perf_counter() - start_time
    pipeline_runtime = runtime_collector.finish_run(total_seconds, stats)

    logger.info("Benchmark complete")
    logger.info(
        "Documents/min: %.2f | Total seconds: %.2f",
        pipeline_runtime.throughput_docs_per_minute,
        pipeline_runtime.total_seconds,
    )

    if pipeline_runtime.gpu_peak_mb is not None:
        logger.info("Peak GPU memory (MB): %.2f", pipeline_runtime.gpu_peak_mb)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "recorded_at": datetime.now().isoformat(),
            "benchmark": pipeline_runtime_to_dict(pipeline_runtime),
        }
        output_path.write_text(json.dumps(payload, indent=2))
        logger.info("Wrote benchmark metrics to %s", output_path)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark the unified ETL pipeline",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=RAW_DIR,
        help="Directory containing input documents",
    )
    parser.add_argument(
        "--pattern",
        default="*.*",
        help="File glob to select documents",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Where to save benchmark JSON (set to '' to disable saving)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    output_path = Path(args.output) if args.output else None
    run_benchmark(args.input_dir, args.pattern, output_path)


if __name__ == "__main__":
    main()
