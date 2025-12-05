#!/usr/bin/env python3
# etl_pipeline/run_etl.py
"""
Main ETL runner script.

Processes documents from data/raw/, generates:
1. Vector store entries (data/vectordb/)
2. Fine-tuning dataset (data/ft/)
"""
import json
import logging
from datetime import datetime
from pathlib import Path

from liquid_shared import (
    DATA_DIR,
    EmbeddingService,
    VectorStore,
)
from pypdf import PdfReader

from .graph_etl import ETLOutput, run_etl_pipeline_sync

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Directories
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FT_DIR = DATA_DIR / "ft"
VDB_DIR = DATA_DIR / "vectordb"

# Ensure directories exist
for dir_path in [RAW_DIR, PROCESSED_DIR, FT_DIR, VDB_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


def load_text_from_pdf(path: Path) -> str:
    """Extract text from PDF."""
    reader = PdfReader(str(path))
    texts = []
    for page in reader.pages:
        try:
            text = page.extract_text()
            if text:
                texts.append(text)
        except Exception as e:
            logger.warning(f"Could not extract page: {e}")
    return "\n\n".join(texts)


def load_text_from_file(path: Path) -> str:
    """Load text from file based on extension."""
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return load_text_from_pdf(path)
    elif suffix in {".txt", ".md", ".rst"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    else:
        logger.warning(f"Unsupported file type: {suffix}")
        return path.read_text(encoding="utf-8", errors="ignore")


def save_ft_dataset(
    samples: list,
    output_path: Path,
    format: str = "jsonl"
) -> None:
    """Save fine-tuning samples to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "jsonl":
        with output_path.open("w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample.model_dump(), ensure_ascii=False) + "\n")
    elif format == "json":
        with output_path.open("w", encoding="utf-8") as f:
            json.dump([s.model_dump() for s in samples], f, ensure_ascii=False, indent=2)
    else:
        raise ValueError(f"Unknown format: {format}")

    logger.info(f"Saved {len(samples)} samples to {output_path}")


def process_document(
    path: Path,
    embedder: EmbeddingService,
    vstore: VectorStore,
) -> ETLOutput | None:
    """
    Process a single document through the ETL pipeline.
    
    Args:
        path: Path to document
        embedder: Embedding service for vectorization
        vstore: Vector store for indexing
        
    Returns:
        ETLOutput or None if processing failed
    """
    logger.info(f"Processing: {path}")

    try:
        # Load document text
        raw_text = load_text_from_file(path)
        if not raw_text.strip():
            logger.warning(f"Empty document: {path}")
            return None

        logger.info(f"Loaded {len(raw_text)} characters from {path.name}")

        # Run ETL pipeline
        result = run_etl_pipeline_sync(str(path), raw_text)

        # Store in vector database
        if result.chunks and result.summaries:
            ids = [c.id for c in result.chunks]

            # Use embedding-optimized text for vectors
            summary_by_chunk = {s.chunk_id: s for s in result.summaries}
            docs = []
            for chunk in result.chunks:
                summary = summary_by_chunk.get(chunk.id)
                docs.append(summary.embedding_text if summary else chunk.text)

            # Generate embeddings
            embeddings = embedder.encode(docs, show_progress=True)

            # Prepare metadata
            metadatas = []
            for chunk in result.chunks:
                summary = summary_by_chunk.get(chunk.id)
                metadatas.append({
                    "source": chunk.source_path,
                    "chunk_index": chunk.chunk_index,
                    "section_title": chunk.section_title or "",
                    "importance": chunk.importance,
                    "tags": ",".join(chunk.tags),
                    "entities": ",".join(chunk.entities),
                    "summary": summary.summary if summary else "",
                })

            # Add to vector store
            vstore.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=[c.text for c in result.chunks],
            )

            logger.info(f"Indexed {len(ids)} chunks to vector store")

        # Save fine-tuning dataset
        if result.ft_samples:
            ft_path = FT_DIR / f"{path.stem}.jsonl"
            save_ft_dataset(result.ft_samples, ft_path)

        # Save processed summary
        summary_path = PROCESSED_DIR / f"{path.stem}_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump({
                "source": str(path),
                "processed_at": datetime.now().isoformat(),
                "stats": result.stats,
                "num_chunks": len(result.chunks),
                "num_summaries": len(result.summaries),
                "num_qa_pairs": len(result.qa_pairs),
                "num_ft_samples": len(result.ft_samples),
            }, f, indent=2)

        return result

    except Exception as e:
        logger.error(f"Failed to process {path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_etl(
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    file_pattern: str = "*.*",
) -> dict:
    """
    Run ETL on all documents in a directory.
    
    Args:
        input_dir: Directory with source documents (default: data/raw/)
        output_dir: Output directory (default: data/)
        file_pattern: Glob pattern for files
        
    Returns:
        Summary statistics
    """
    input_dir = input_dir or RAW_DIR

    # Initialize services
    logger.info("Initializing embedding service...")
    embedder = EmbeddingService()

    logger.info("Initializing vector store...")
    vstore = VectorStore(VDB_DIR)

    # Find documents
    supported_extensions = {".pdf", ".txt", ".md", ".rst"}
    documents = [
        p for p in input_dir.glob(file_pattern)
        if p.is_file() and p.suffix.lower() in supported_extensions
    ]

    if not documents:
        logger.warning(f"No documents found in {input_dir}")
        return {"documents_processed": 0}

    logger.info(f"Found {len(documents)} documents to process")

    # Process documents
    stats = {
        "documents_processed": 0,
        "documents_failed": 0,
        "total_chunks": 0,
        "total_qa_pairs": 0,
        "total_ft_samples": 0,
    }

    for doc_path in documents:
        result = process_document(doc_path, embedder, vstore)

        if result:
            stats["documents_processed"] += 1
            stats["total_chunks"] += len(result.chunks)
            stats["total_qa_pairs"] += len(result.qa_pairs)
            stats["total_ft_samples"] += len(result.ft_samples)
        else:
            stats["documents_failed"] += 1

    # Persist vector store
    vstore.persist()

    # Log summary
    logger.info("=" * 50)
    logger.info("ETL Complete!")
    logger.info(f"Documents processed: {stats['documents_processed']}")
    logger.info(f"Documents failed: {stats['documents_failed']}")
    logger.info(f"Total chunks: {stats['total_chunks']}")
    logger.info(f"Total QA pairs: {stats['total_qa_pairs']}")
    logger.info(f"Total FT samples: {stats['total_ft_samples']}")
    logger.info(f"Vector store: {vstore.count()} documents")
    logger.info("=" * 50)

    return stats


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ETL pipeline on documents"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=RAW_DIR,
        help="Input directory with documents"
    )
    parser.add_argument(
        "--pattern",
        default="*.*",
        help="File glob pattern"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    run_etl(
        input_dir=args.input_dir,
        file_pattern=args.pattern,
    )


if __name__ == "__main__":
    main()
