"""
High-Volume ETL Pipeline Graph - Creates up to 4000 chunks, skips QA generation.

Modified pipeline:
1. High-volume chunking (small chunks, ~400 chars each)
2. Metadata extraction
3. Classification
4. Taxonomy extraction
5. Summary generation
6. OUTPUT: Vector store only (no fine-tuning dataset)
"""
import logging
from dataclasses import dataclass, field
from typing import Iterable

from liquid_shared import (
    Chunk,
    ChunkSummary,
    FineTuneSample,
    QAPair,
)
from pydantic_graph.beta import GraphBuilder, StepContext

from .agents import (
    extract_metadata,
    classify_chunks,
    extract_taxonomies,
    generate_summaries,
)
from .config_high_volume import fallback_chunk_high_volume, HIGH_VOLUME_CHUNKING

logger = logging.getLogger(__name__)


@dataclass
class ETLState:
    """State object for high-volume ETL pipeline."""
    # Input
    source_path: str
    raw_text: str = ""

    # Intermediate results
    chunk_texts: list[str] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    summaries: list[ChunkSummary] = field(default_factory=list)
    qa_pairs: list[QAPair] = field(default_factory=list)  # Empty
    validated_qa_pairs: list[QAPair] = field(default_factory=list)  # Empty

    # Output
    ft_samples: list[FineTuneSample] = field(default_factory=list)  # Empty

    # Tracking
    errors: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


@dataclass
class ETLOutput:
    """Final output from the ETL pipeline."""
    chunks: list[Chunk]
    summaries: list[ChunkSummary]
    qa_pairs: list[QAPair]
    ft_samples: list[FineTuneSample]
    stats: dict


def build_etl_graph_high_volume():
    """
    Build high-volume ETL pipeline graph.

    Pipeline:
        start
          │
          ▼
        chunk (high-volume, deterministic)
          │
          ▼
       metadata
          │
          ▼
      classification
          │
          ▼
       taxonomy
          │
          ▼
       summary
          │
          ▼
        end (NO QA generation)
    """
    g = GraphBuilder(state_type=ETLState, output_type=ETLOutput)

    # Step 1: High-Volume Chunking (deterministic, no agent)
    @g.step
    async def chunk_document_step(ctx: StepContext[ETLState, None, None]) -> list[str]:
        """Split document into many small chunks (up to 4000)."""
        logger.info(f"High-volume chunking: {ctx.state.source_path}")

        try:
            # Use deterministic chunking for consistency and speed
            chunks = fallback_chunk_high_volume(ctx.state.raw_text, HIGH_VOLUME_CHUNKING)
            ctx.state.chunk_texts = chunks
            ctx.state.stats["chunking_strategy"] = "high_volume_deterministic"
            ctx.state.stats["num_chunks"] = len(chunks)
            ctx.state.stats["avg_chunk_size"] = sum(len(c) for c in chunks) / len(chunks) if chunks else 0
            logger.info(f"Created {len(chunks)} chunks (avg {ctx.state.stats['avg_chunk_size']:.0f} chars)")
            return chunks
        except Exception as e:
            logger.error(f"High-volume chunking failed: {e}")
            ctx.state.errors.append(f"Chunking failed: {e}")
            raise

    # Step 2: Metadata Extraction
    @g.step
    async def extract_metadata_step(ctx: StepContext[ETLState, None, list[str]]) -> list[Chunk]:
        """Extract metadata (including keywords) from chunks."""
        logger.info(f"Extracting metadata for {len(ctx.inputs)} chunks...")

        try:
            # Process in batches to avoid overwhelming the model
            BATCH_SIZE = 100
            all_chunks = []

            for i in range(0, len(ctx.inputs), BATCH_SIZE):
                batch = ctx.inputs[i:i+BATCH_SIZE]
                logger.info(f"Processing metadata batch {i//BATCH_SIZE + 1}/{(len(ctx.inputs)-1)//BATCH_SIZE + 1}")
                batch_chunks = await extract_metadata(batch, ctx.state.source_path)
                all_chunks.extend(batch_chunks)

            ctx.state.chunks = all_chunks
            logger.info(f"Extracted metadata for {len(all_chunks)} chunks")
            return all_chunks
        except Exception as e:
            # If metadata fails, create minimal chunks
            logger.warning(f"Metadata extraction failed, using minimal chunks: {e}")
            ctx.state.errors.append(f"Metadata extraction failed: {e}")

            minimal_chunks = [
                Chunk(
                    id=f"{ctx.state.source_path}_{i}",
                    text=text,
                    source_path=ctx.state.source_path,
                    chunk_index=i,
                    section_title="",
                    importance=5,
                    tags=[],
                    entities=[],
                )
                for i, text in enumerate(ctx.inputs)
            ]
            ctx.state.chunks = minimal_chunks
            return minimal_chunks

    # Step 3: Classification
    @g.step
    async def classify_chunks_step(ctx: StepContext[ETLState, None, list[Chunk]]) -> list[Chunk]:
        """Classify chunks into categories."""
        logger.info(f"Classifying {len(ctx.inputs)} chunks...")

        try:
            # Process in batches
            BATCH_SIZE = 50

            for i in range(0, len(ctx.state.chunks), BATCH_SIZE):
                batch_chunks = ctx.state.chunks[i:i+BATCH_SIZE]
                logger.info(f"Classifying batch {i//BATCH_SIZE + 1}/{(len(ctx.state.chunks)-1)//BATCH_SIZE + 1}")

                classifications = await classify_chunks(batch_chunks)

                # Update chunks with classification results
                for classification in classifications.classifications:
                    chunk_idx = i + classification.chunk_index
                    if chunk_idx < len(ctx.state.chunks):
                        chunk = ctx.state.chunks[chunk_idx]
                        chunk.categories = classification.categories

            logger.info(f"Classified {len(ctx.state.chunks)} chunks")
            return ctx.state.chunks
        except Exception as e:
            ctx.state.errors.append(f"Classification failed: {e}")
            logger.warning(f"Classification failed, continuing without categories: {e}")
            return ctx.inputs

    # Step 4: Taxonomy Extraction
    @g.step
    async def extract_taxonomy_step(ctx: StepContext[ETLState, None, list[Chunk]]) -> list[Chunk]:
        """Extract hierarchical taxonomies from chunks."""
        logger.info(f"Extracting taxonomies for {len(ctx.inputs)} chunks...")

        try:
            # Process in batches
            BATCH_SIZE = 50

            for i in range(0, len(ctx.state.chunks), BATCH_SIZE):
                batch_chunks = ctx.state.chunks[i:i+BATCH_SIZE]
                logger.info(f"Extracting taxonomies batch {i//BATCH_SIZE + 1}/{(len(ctx.state.chunks)-1)//BATCH_SIZE + 1}")

                taxonomies = await extract_taxonomies(batch_chunks)

                # Update chunks with taxonomy results
                for taxonomy in taxonomies.taxonomies:
                    chunk_idx = i + taxonomy.chunk_index
                    if chunk_idx < len(ctx.state.chunks):
                        chunk = ctx.state.chunks[chunk_idx]
                        chunk.taxonomy = taxonomy.taxonomy

            logger.info(f"Extracted taxonomies for {len(ctx.state.chunks)} chunks")
            return ctx.state.chunks
        except Exception as e:
            ctx.state.errors.append(f"Taxonomy extraction failed: {e}")
            logger.warning(f"Taxonomy extraction failed, continuing without taxonomies: {e}")
            return ctx.inputs

    # Step 5: Summary Generation
    @g.step
    async def generate_summaries_step(ctx: StepContext[ETLState, None, list[Chunk]]) -> list[ChunkSummary]:
        """Generate summaries for chunks."""
        logger.info(f"Generating summaries for {len(ctx.state.chunks)} chunks...")

        try:
            # Process in batches
            BATCH_SIZE = 50
            all_summaries = []

            for i in range(0, len(ctx.state.chunks), BATCH_SIZE):
                batch_chunks = ctx.state.chunks[i:i+BATCH_SIZE]
                logger.info(f"Generating summaries batch {i//BATCH_SIZE + 1}/{(len(ctx.state.chunks)-1)//BATCH_SIZE + 1}")

                summaries = await generate_summaries(batch_chunks)
                all_summaries.extend(summaries)

            ctx.state.summaries = all_summaries
            logger.info(f"Generated {len(all_summaries)} summaries")
            return all_summaries
        except Exception as e:
            ctx.state.errors.append(f"Summary generation failed: {e}")
            logger.warning(f"Summary generation failed, using chunk text as fallback: {e}")

            # Create minimal summaries
            fallback_summaries = [
                ChunkSummary(
                    chunk_id=chunk.id,
                    summary=chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                    embedding_text=chunk.text,
                )
                for chunk in ctx.state.chunks
            ]
            ctx.state.summaries = fallback_summaries
            return fallback_summaries

    # NO QA generation steps - skip directly to output

    # Final step: Build output
    @g.step
    async def build_output_step(ctx: StepContext[ETLState, None, list[ChunkSummary]]) -> ETLOutput:
        """Build final output (no fine-tuning dataset)."""
        ctx.state.stats["qa_generation"] = "skipped"
        ctx.state.stats["ft_samples"] = 0

        return ETLOutput(
            chunks=ctx.state.chunks,
            summaries=ctx.state.summaries,
            qa_pairs=[],  # Empty - no QA generation
            ft_samples=[],  # Empty - no fine-tuning dataset
            stats=ctx.state.stats,
        )

    # Wire the graph
    g.edge_from(chunk_document_step).to(extract_metadata_step)
    g.edge_from(extract_metadata_step).to(classify_chunks_step)
    g.edge_from(classify_chunks_step).to(extract_taxonomy_step)
    g.edge_from(extract_taxonomy_step).to(generate_summaries_step)
    g.edge_from(generate_summaries_step).to(build_output_step)

    return g.build()


def run_etl_pipeline_sync(source_path: str, raw_text: str) -> ETLOutput:
    """Synchronous wrapper for high-volume ETL pipeline."""
    import asyncio

    graph = build_etl_graph_high_volume()

    state = ETLState(
        source_path=source_path,
        raw_text=raw_text,
    )

    result = asyncio.run(graph.run(state))
    return result.output
