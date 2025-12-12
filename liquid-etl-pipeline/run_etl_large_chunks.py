#!/usr/bin/env python3
"""
ETL runner with large chunks (4000 tokens max) and no QA generation.

Usage:
    python run_etl_large_chunks.py
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from time import perf_counter

from liquid_shared import (
    DATA_DIR,
    EmbeddingService,
    VectorStore,
    Chunk,
    ChunkSummary,
)
from pypdf import PdfReader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Directories
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
VDB_DIR = DATA_DIR / "vectordb"

for dir_path in [RAW_DIR, PROCESSED_DIR, VDB_DIR]:
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


def chunk_text_large(text: str, max_tokens: int = 4000) -> list[str]:
    """
    Split text into large chunks (~4000 tokens max).

    Approximate: 1 token ≈ 4 characters
    So 4000 tokens ≈ 16,000 characters
    """
    max_chars = max_tokens * 4  # 16,000 chars
    overlap_chars = 400  # ~100 token overlap

    if not text:
        return []

    chunks = []
    paragraphs = text.split("\n\n")

    current_chunk = []
    current_length = 0

    for para in paragraphs:
        para_len = len(para)

        # If adding this paragraph exceeds max, save current chunk
        if current_length + para_len > max_chars and current_chunk:
            chunk_text = "\n\n".join(current_chunk).strip()
            if chunk_text:
                chunks.append(chunk_text)

            # Start new chunk with overlap (last few paragraphs)
            overlap_paras = current_chunk[-2:] if len(current_chunk) >= 2 else current_chunk
            current_chunk = overlap_paras
            current_length = sum(len(p) for p in current_chunk)

        current_chunk.append(para)
        current_length += para_len + 2  # +2 for "\n\n"

    # Add remaining chunk
    if current_chunk:
        chunk_text = "\n\n".join(current_chunk).strip()
        if chunk_text:
            chunks.append(chunk_text)

    logger.info(f"Created {len(chunks)} chunks (avg {sum(len(c) for c in chunks) / len(chunks):.0f} chars)")
    return chunks


def process_document_large_chunks(
    path: Path,
    embedder: EmbeddingService,
    vstore: VectorStore,
) -> dict:
    """
    Process document with large chunks (4000 tokens max).

    Pipeline:
    1. Load document
    2. Chunk into large pieces (4000 tokens max)
    3. Create minimal metadata
    4. Generate embeddings
    5. Store in vector database

    NO QA generation.
    """
    logger.info(f"Processing: {path}")

    try:
        # Load document text
        raw_text = load_text_from_file(path)
        if not raw_text.strip():
            logger.warning(f"Empty document: {path}")
            return {"success": False, "error": "empty"}

        logger.info(f"Loaded {len(raw_text)} characters from {path.name}")

        # Chunk into large pieces
        chunk_texts = chunk_text_large(raw_text, max_tokens=4000)

        if not chunk_texts:
            logger.warning(f"No chunks created for {path}")
            return {"success": False, "error": "no_chunks"}

        # Create minimal Chunk objects
        chunks = []
        for i, text in enumerate(chunk_texts):
            chunk = Chunk(
                id=f"{path.stem}_{i}",
                text=text,
                source_path=str(path),
                chunk_index=i,
                section_title=f"Section {i+1}",
                importance=5,  # Default importance
                tags=[],
                entities=[],
            )
            chunks.append(chunk)

        # Generate embeddings (use chunk text directly)
        docs = [c.text for c in chunks]
        embeddings = embedder.encode(docs, show_progress=True)

        # Prepare metadata
        metadatas = []
        for chunk in chunks:
            metadatas.append({
                "source": chunk.source_path,
                "chunk_index": chunk.chunk_index,
                "section_title": chunk.section_title,
                "importance": chunk.importance,
                "tags": "",
                "entities": "",
                "summary": "",
            })

        # Add to vector store
        ids = [c.id for c in chunks]
        vstore.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=docs,
        )

        logger.info(f"Indexed {len(ids)} chunks to vector store")

        # Save processing summary
        summary_path = PROCESSED_DIR / f"{path.stem}_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump({
                "source": str(path),
                "processed_at": datetime.now().isoformat(),
                "num_chunks": len(chunks),
                "avg_chunk_size": sum(len(c.text) for c in chunks) / len(chunks),
                "max_chunk_size": max(len(c.text) for c in chunks),
                "min_chunk_size": min(len(c.text) for c in chunks),
                "total_chars": len(raw_text),
                "qa_pairs": 0,  # No QA generation
            }, f, indent=2)

        return {
            "success": True,
            "num_chunks": len(chunks),
            "total_chars": len(raw_text),
        }

    except Exception as e:
        logger.error(f"Failed to process {path}: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def main():
    """Run ETL with large chunks and no QA generation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="ETL with large chunks (4000 tokens max), no QA generation"
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
        "--clear-db",
        action="store_true",
        help="Clear vector database before processing"
    )

    args = parser.parse_args()

    # Initialize services
    logger.info("Initializing embedding service...")
    embedder = EmbeddingService()

    logger.info("Initializing vector store...")
    vstore = VectorStore(VDB_DIR)

    if args.clear_db:
        logger.info("Clearing vector database...")
        try:
            # Delete and recreate
            import shutil
            if VDB_DIR.exists():
                shutil.rmtree(VDB_DIR)
            VDB_DIR.mkdir(parents=True, exist_ok=True)
            vstore = VectorStore(VDB_DIR)
            logger.info("Vector database cleared")
        except Exception as e:
            logger.warning(f"Could not clear database: {e}")

    # Find documents
    supported_extensions = {".pdf", ".txt", ".md", ".rst"}
    documents = [
        p for p in args.input_dir.glob(args.pattern)
        if p.is_file() and p.suffix.lower() in supported_extensions
    ]

    if not documents:
        logger.warning(f"No documents found in {args.input_dir}")
        return

    logger.info(f"Found {len(documents)} documents to process")
    logger.info("Configuration: 4000 tokens/chunk max, NO QA generation")

    # Process documents
    stats = {
        "documents_processed": 0,
        "documents_failed": 0,
        "total_chunks": 0,
    }

    for doc_path in documents:
        start_time = perf_counter()
        result = process_document_large_chunks(doc_path, embedder, vstore)
        elapsed = perf_counter() - start_time

        if result.get("success"):
            stats["documents_processed"] += 1
            stats["total_chunks"] += result.get("num_chunks", 0)
            logger.info(f"Processed {doc_path.name} in {elapsed:.2f}s")
        else:
            stats["documents_failed"] += 1

    # Persist vector store
    vstore.persist()

    # Log summary
    logger.info("=" * 60)
    logger.info("ETL Complete (Large Chunks Mode)!")
    logger.info(f"Documents processed: {stats['documents_processed']}")
    logger.info(f"Documents failed: {stats['documents_failed']}")
    logger.info(f"Total chunks: {stats['total_chunks']}")
    logger.info(f"Vector store: {vstore.count()} documents")
    logger.info(f"Configuration: 4000 tokens/chunk max (~16,000 chars)")
    logger.info(f"QA pairs generated: 0 (skipped)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
