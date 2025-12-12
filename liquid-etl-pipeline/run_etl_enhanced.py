#!/usr/bin/env python3
"""
Enhanced ETL: Large chunks (4000 tokens) + Full metadata + Domain-specific QA.

Features:
- 4000 tokens per chunk max (~16,000 chars)
- Full metadata extraction (tags, entities, importance)
- Classification into domain categories
- Taxonomy extraction
- Summaries for embeddings
- ENHANCED QA: 5-10 QA pairs per chunk with specialized types:
  * NER questions (named entities)
  * Technical questions (mechanisms, how-to)
  * Domain questions (risks, mitigations)
  * Conceptual questions (definitions, relationships)

Usage:
    python run_etl_enhanced.py [--clear-db]
"""
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

# Add liquid-etl-pipeline to path
sys.path.insert(0, str(Path(__file__).parent))

from liquid_shared import (
    DATA_DIR,
    EmbeddingService,
    VectorStore,
    Chunk,
)
from pypdf import PdfReader

# Import agents
from etl_pipeline.agents import (
    extract_metadata,
    classify_chunks,
    extract_taxonomies,
    generate_summaries,
)
from etl_pipeline.agents.qa_enhanced import generate_enhanced_qa_pairs

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

            # Start new chunk with overlap
            overlap_paras = current_chunk[-2:] if len(current_chunk) >= 2 else current_chunk
            current_chunk = overlap_paras
            current_length = sum(len(p) for p in current_chunk)

        current_chunk.append(para)
        current_length += para_len + 2

    # Add remaining chunk
    if current_chunk:
        chunk_text = "\n\n".join(current_chunk).strip()
        if chunk_text:
            chunks.append(chunk_text)

    return chunks


async def process_document_enhanced(
    path: Path,
    embedder: EmbeddingService,
    vstore: VectorStore,
) -> dict:
    """
    Process document with enhanced pipeline:
    1. Large chunks (4000 tokens max)
    2. Full metadata extraction
    3. Classification
    4. Taxonomy
    5. Summaries
    6. Enhanced QA (5-10 pairs per chunk, domain-specific)
    """
    logger.info(f"Processing: {path.name}")

    try:
        # 1. Load document
        raw_text = load_text_from_file(path)
        if not raw_text.strip():
            return {"success": False, "error": "empty"}

        logger.info(f"Loaded {len(raw_text)} characters")

        # 2. Chunk into large pieces
        chunk_texts = chunk_text_large(raw_text, max_tokens=4000)
        logger.info(f"Created {len(chunk_texts)} chunks")

        if not chunk_texts:
            return {"success": False, "error": "no_chunks"}

        # 3. Metadata extraction
        logger.info("Extracting metadata...")
        chunks = await extract_metadata(chunk_texts, str(path))
        logger.info(f"Extracted metadata for {len(chunks)} chunks")

        # 4. Classification
        logger.info("Classifying chunks...")
        try:
            classifications = await classify_chunks(chunks)
            for classification in classifications.classifications:
                if classification.chunk_index < len(chunks):
                    chunks[classification.chunk_index].categories = classification.categories
            logger.info(f"Classified {len(chunks)} chunks")
        except Exception as e:
            logger.warning(f"Classification failed: {e}")

        # 5. Taxonomy extraction
        logger.info("Extracting taxonomies...")
        try:
            taxonomies = await extract_taxonomies(chunks)
            for taxonomy in taxonomies.taxonomies:
                if taxonomy.chunk_index < len(chunks):
                    chunks[taxonomy.chunk_index].taxonomy = taxonomy.taxonomy
            logger.info(f"Extracted taxonomies for {len(chunks)} chunks")
        except Exception as e:
            logger.warning(f"Taxonomy extraction failed: {e}")

        # 6. Summary generation
        logger.info("Generating summaries...")
        summaries = await generate_summaries(chunks)
        logger.info(f"Generated {len(summaries)} summaries")

        # 7. ENHANCED QA generation (5-10 per chunk, domain-specific)
        logger.info("Generating enhanced QA pairs (5-10 per chunk)...")
        qa_pairs = await generate_enhanced_qa_pairs(chunks)
        logger.info(f"Generated {len(qa_pairs)} QA pairs ({len(qa_pairs)/len(chunks):.1f} per chunk)")

        # Count QA by type
        qa_by_type = {}
        for qa in qa_pairs:
            q_type = qa.metadata.get("question_type", "unknown")
            qa_by_type[q_type] = qa_by_type.get(q_type, 0) + 1
        logger.info(f"QA breakdown: {qa_by_type}")

        # 8. Store in vector database
        summary_by_chunk = {s.chunk_id: s for s in summaries}
        docs = []
        for chunk in chunks:
            summary = summary_by_chunk.get(chunk.id)
            docs.append(summary.embedding_text if summary else chunk.text)

        embeddings = embedder.encode(docs, show_progress=True)

        metadatas = []
        for chunk in chunks:
            summary = summary_by_chunk.get(chunk.id)
            metadatas.append({
                "source": chunk.source_path,
                "chunk_index": chunk.chunk_index,
                "section_title": chunk.section_title or "",
                "importance": chunk.importance,
                "tags": ",".join(chunk.tags),
                "entities": ",".join(chunk.entities),
                "summary": summary.summary if summary else "",
                "categories": ",".join(chunk.categories) if chunk.categories else "",
                "taxonomy": ",".join(chunk.taxonomy) if chunk.taxonomy else "",
            })

        ids = [c.id for c in chunks]
        vstore.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=[c.text for c in chunks],
        )
        logger.info(f"Indexed {len(ids)} chunks to vector store")

        # 9. Save fine-tuning dataset
        if qa_pairs:
            from liquid_shared import FineTuneSample

            ft_samples = []
            for qa in qa_pairs:
                # Create instruction-format sample
                ft_sample = FineTuneSample(
                    instruction=qa.question,
                    input="",
                    output=qa.answer,
                    metadata={
                        "source": str(path),
                        "question_type": qa.metadata.get("question_type", "unknown"),
                        "chunk_id": qa.metadata.get("chunk_id", ""),
                        "quality_score": qa.quality_score,
                    }
                )
                ft_samples.append(ft_sample)

            ft_path = FT_DIR / f"{path.stem}_enhanced.jsonl"
            with ft_path.open("w", encoding="utf-8") as f:
                for sample in ft_samples:
                    f.write(json.dumps(sample.model_dump(), ensure_ascii=False) + "\n")
            logger.info(f"Saved {len(ft_samples)} fine-tuning samples")

        # 10. Save summary
        summary_path = PROCESSED_DIR / f"{path.stem}_enhanced_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump({
                "source": str(path),
                "processed_at": datetime.now().isoformat(),
                "num_chunks": len(chunks),
                "avg_chunk_size": sum(len(c.text) for c in chunks) / len(chunks),
                "num_qa_pairs": len(qa_pairs),
                "qa_per_chunk": len(qa_pairs) / len(chunks) if chunks else 0,
                "qa_by_type": qa_by_type,
                "total_chars": len(raw_text),
            }, f, indent=2)

        return {
            "success": True,
            "num_chunks": len(chunks),
            "num_qa_pairs": len(qa_pairs),
            "qa_by_type": qa_by_type,
        }

    except Exception as e:
        logger.error(f"Failed to process {path}: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def main():
    """Run enhanced ETL pipeline."""
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(
        description="Enhanced ETL: Large chunks + Full metadata + Domain QA"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=RAW_DIR,
        help="Input directory"
    )
    parser.add_argument(
        "--clear-db",
        action="store_true",
        help="Clear vector database first"
    )

    args = parser.parse_args()

    # Initialize services
    logger.info("Initializing services...")
    embedder = EmbeddingService()
    vstore = VectorStore(VDB_DIR)

    if args.clear_db:
        logger.info("Clearing vector database...")
        import shutil
        if VDB_DIR.exists():
            shutil.rmtree(VDB_DIR)
        VDB_DIR.mkdir(parents=True, exist_ok=True)
        vstore = VectorStore(VDB_DIR)

    # Find documents
    supported_extensions = {".pdf", ".txt", ".md", ".rst"}
    documents = [
        p for p in args.input_dir.glob("*.*")
        if p.is_file() and p.suffix.lower() in supported_extensions
    ]

    if not documents:
        logger.warning(f"No documents found in {args.input_dir}")
        return

    logger.info(f"Found {len(documents)} documents")
    logger.info("Configuration:")
    logger.info("  - Chunk size: 4000 tokens max (~16,000 chars)")
    logger.info("  - Metadata: FULL (tags, entities, importance)")
    logger.info("  - Classification: YES")
    logger.info("  - Taxonomy: YES")
    logger.info("  - Summaries: YES")
    logger.info("  - QA: ENHANCED (5-10 per chunk, domain-specific)")

    # Process documents
    stats = {
        "documents_processed": 0,
        "documents_failed": 0,
        "total_chunks": 0,
        "total_qa_pairs": 0,
        "qa_by_type": {},
    }

    for doc_path in documents:
        start_time = perf_counter()
        result = asyncio.run(process_document_enhanced(doc_path, embedder, vstore))
        elapsed = perf_counter() - start_time

        if result.get("success"):
            stats["documents_processed"] += 1
            stats["total_chunks"] += result.get("num_chunks", 0)
            stats["total_qa_pairs"] += result.get("num_qa_pairs", 0)

            # Aggregate QA by type
            for q_type, count in result.get("qa_by_type", {}).items():
                stats["qa_by_type"][q_type] = stats["qa_by_type"].get(q_type, 0) + count

            logger.info(f"✓ {doc_path.name} ({elapsed:.1f}s)")
        else:
            stats["documents_failed"] += 1
            logger.error(f"✗ {doc_path.name}")

    # Persist
    vstore.persist()

    # Summary
    logger.info("=" * 70)
    logger.info("ENHANCED ETL COMPLETE!")
    logger.info(f"Documents processed: {stats['documents_processed']}")
    logger.info(f"Documents failed: {stats['documents_failed']}")
    logger.info(f"Total chunks: {stats['total_chunks']}")
    logger.info(f"Total QA pairs: {stats['total_qa_pairs']}")
    if stats['total_chunks'] > 0:
        logger.info(f"QA pairs per chunk: {stats['total_qa_pairs'] / stats['total_chunks']:.1f}")
    logger.info(f"QA breakdown by type: {stats['qa_by_type']}")
    logger.info(f"Vector store: {vstore.count()} documents")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
