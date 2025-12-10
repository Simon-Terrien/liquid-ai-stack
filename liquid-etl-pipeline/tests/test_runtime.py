from pathlib import Path
from types import SimpleNamespace

import pytest

from etl_pipeline.experiments.runtime import RuntimeCollector, pipeline_runtime_to_dict


def make_result(num_chunks: int, num_qa_pairs: int, num_ft_samples: int, qa_pass_rate: float | None):
    return SimpleNamespace(
        chunks=[object()] * num_chunks,
        qa_pairs=[object()] * num_qa_pairs,
        ft_samples=[object()] * num_ft_samples,
        stats={"qa_pass_rate": qa_pass_rate} if qa_pass_rate is not None else {},
    )


def test_runtime_collector_records_and_summarizes(monkeypatch):
    collector = RuntimeCollector()
    # avoid touching GPU in tests
    collector._gpu_available = False

    doc_one = make_result(num_chunks=2, num_qa_pairs=1, num_ft_samples=3, qa_pass_rate=0.5)
    doc_two = make_result(num_chunks=0, num_qa_pairs=0, num_ft_samples=0, qa_pass_rate=None)

    collector.record_document(Path("doc1.txt"), doc_one, wall_seconds=1.5)
    collector.record_document(Path("doc2.txt"), doc_two, wall_seconds=0.5)

    runtime = collector.finish_run(total_seconds=3.0, aggregate_stats={"documents_processed": 2})

    assert runtime.total_seconds == pytest.approx(3.0)
    # 2 docs over 3 seconds = 40 docs/min
    assert runtime.throughput_docs_per_minute == pytest.approx(40.0)
    assert runtime.documents[0].path.endswith("doc1.txt")
    assert runtime.documents[0].num_chunks == 2
    assert runtime.documents[0].qa_pass_rate == 0.5
    assert runtime.documents[1].num_chunks == 0
    assert runtime.documents[1].qa_pass_rate is None
    assert runtime.gpu_peak_mb is None


def test_pipeline_runtime_to_dict_serializes_documents():
    collector = RuntimeCollector()
    collector._gpu_available = False

    result = make_result(num_chunks=1, num_qa_pairs=2, num_ft_samples=1, qa_pass_rate=0.8)
    collector.record_document(Path("doc.txt"), result, wall_seconds=2.0)
    runtime = collector.finish_run(total_seconds=2.0, aggregate_stats={"documents_processed": 1})

    payload = pipeline_runtime_to_dict(runtime)

    assert payload["total_seconds"] == pytest.approx(2.0)
    assert payload["throughput_docs_per_minute"] == pytest.approx(30.0)
    assert payload["gpu_peak_mb"] is None
    assert payload["aggregate_stats"] == {"documents_processed": 1}
    assert payload["documents"][0]["num_qa_pairs"] == 2
    assert payload["documents"][0]["qa_pass_rate"] == 0.8
