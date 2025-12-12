"""
High-volume chunking configuration for ETL pipeline.

Creates up to 4000 small chunks instead of generating QA pairs.
Optimized for maximum retrieval coverage and granularity.
"""

# Chunking configuration
HIGH_VOLUME_CHUNKING = {
    "target_chunk_size": 400,  # ~100 tokens (smaller chunks)
    "max_chunk_size": 800,     # ~200 tokens max
    "min_chunk_size": 200,     # ~50 tokens min
    "overlap": 100,            # 100 char overlap for context preservation
    "max_total_chunks": 4000,  # Hard limit
}

# Pipeline configuration
PIPELINE_CONFIG = {
    "enable_chunking": True,
    "enable_metadata": True,     # Keep metadata extraction
    "enable_classification": True,  # Keep classification
    "enable_taxonomy": True,     # Keep taxonomy
    "enable_summaries": True,    # Keep summaries for embeddings
    "enable_qa_generation": False,  # SKIP QA generation
    "enable_qa_validation": False,  # SKIP QA validation
}

# Fallback chunking for high-volume mode
def fallback_chunk_high_volume(text: str, config: dict = None) -> list[str]:
    """
    Deterministic high-volume chunking using sliding windows.

    Creates small, overlapping chunks to maximize coverage.
    """
    if config is None:
        config = HIGH_VOLUME_CHUNKING

    chunk_size = config["target_chunk_size"]
    overlap = config["overlap"]
    max_chunks = config["max_total_chunks"]

    if not text:
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len and len(chunks) < max_chunks:
        # Extract chunk
        end = min(start + chunk_size, text_len)

        # Try to break at sentence boundary (. ! ?)
        if end < text_len:
            # Look for sentence boundary in the last 100 chars
            last_section = text[max(start, end-100):end]
            sentence_end = max(
                last_section.rfind('. '),
                last_section.rfind('! '),
                last_section.rfind('? '),
                last_section.rfind('\n\n'),
            )
            if sentence_end > 0:
                end = max(start, end - 100 + sentence_end + 2)

        chunk_text = text[start:end].strip()
        if chunk_text and len(chunk_text) >= config.get("min_chunk_size", 200):
            chunks.append(chunk_text)

        # Move to next chunk with overlap
        start = end - overlap

        # Prevent infinite loop
        if start <= 0 or end >= text_len:
            break

    return chunks[:max_chunks]  # Enforce hard limit
