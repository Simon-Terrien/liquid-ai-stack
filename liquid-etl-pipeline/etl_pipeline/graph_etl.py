# etl_pipeline/graph_etl.py
"""
ETL Pipeline Graph using Pydantic Graph beta API.

Orchestrates the complete ETL workflow:
1. Chunking (semantic document splitting)
2. Metadata extraction (parallel with summaries)
3. Summary generation (parallel with metadata)
4. QA pair generation
5. QA validation
6. Output: Vector store + Fine-tuning dataset
"""
import logging
from dataclasses import dataclass, field

from liquid_shared import (
    Chunk,
    ChunkSummary,
    FineTuneSample,
    QAPair,
)
from pydantic_graph.beta import GraphBuilder, StepContext

from .agents import (
    chunk_document,
    extract_metadata,
    generate_qa_pairs,
    generate_summaries,
    validate_qa_pairs,
)

logger = logging.getLogger(__name__)


@dataclass
class ETLState:
    """
    State object for the ETL pipeline.
    
    Tracks all intermediate results as data flows through the pipeline.
    """
    # Input
    source_path: str
    raw_text: str = ""

    # Intermediate results
    chunk_texts: list[str] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    summaries: list[ChunkSummary] = field(default_factory=list)
    qa_pairs: list[QAPair] = field(default_factory=list)
    validated_qa_pairs: list[QAPair] = field(default_factory=list)

    # Output
    ft_samples: list[FineTuneSample] = field(default_factory=list)

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


def build_etl_graph():
    """
    Build the ETL pipeline graph.
    
    Graph structure:
    
        start
          │
          ▼
        chunk
          │
          ├──────────┬──────────┐
          ▼          ▼          │
       metadata   summary       │
          │          │          │
          └────┬─────┘          │
               ▼                │
              join              │
               │                │
               ▼                │
           generate_qa ◄────────┘
               │
               ▼
          validate_qa
               │
               ▼
         build_ft_data
               │
               ▼
             end
    """
    g = GraphBuilder(state_type=ETLState, output_type=ETLOutput)

    # Step 1: Intelligent Chunking
    @g.step
    async def chunk_document_step(ctx: StepContext[ETLState, None, None]) -> list[str]:
        """Split document into semantic chunks."""
        logger.info(f"Chunking document: {ctx.state.source_path}")

        try:
            result = await chunk_document(ctx.state.raw_text)
            ctx.state.chunk_texts = result.chunks
            ctx.state.stats["num_chunks"] = len(result.chunks)
            logger.info(f"Created {len(result.chunks)} chunks")
            return result.chunks
        except Exception as e:
            ctx.state.errors.append(f"Chunking failed: {e}")
            logger.error(f"Chunking failed: {e}")
            raise

    # Step 2a: Metadata Extraction
    @g.step
    async def extract_metadata_step(ctx: StepContext[ETLState, None, list[str]]) -> list[Chunk]:
        """Extract metadata from chunks."""
        logger.info("Extracting metadata...")

        try:
            chunks = await extract_metadata(
                ctx.inputs,
                ctx.state.source_path
            )
            ctx.state.chunks = chunks
            logger.info(f"Extracted metadata for {len(chunks)} chunks")
            return chunks
        except Exception as e:
            ctx.state.errors.append(f"Metadata extraction failed: {e}")
            logger.error(f"Metadata extraction failed: {e}")
            raise

    # Step 2b: Summary Generation (parallel with metadata)
    @g.step
    async def generate_summaries_step(ctx: StepContext[ETLState, None, list[Chunk]]) -> list[ChunkSummary]:
        """Generate summaries for chunks."""
        logger.info("Generating summaries...")

        try:
            # Wait for chunks to have metadata
            chunks = ctx.state.chunks if ctx.state.chunks else ctx.inputs
            summaries = await generate_summaries(chunks)
            ctx.state.summaries = summaries
            logger.info(f"Generated {len(summaries)} summaries")
            return summaries
        except Exception as e:
            ctx.state.errors.append(f"Summary generation failed: {e}")
            logger.error(f"Summary generation failed: {e}")
            raise

    # Step 3: QA Generation
    @g.step
    async def generate_qa_step(ctx: StepContext[ETLState, None, list[ChunkSummary]]) -> list[QAPair]:
        """Generate QA pairs for fine-tuning."""
        logger.info("Generating QA pairs...")

        try:
            qa_pairs = await generate_qa_pairs(ctx.state.chunks)
            ctx.state.qa_pairs = qa_pairs
            ctx.state.stats["num_qa_generated"] = len(qa_pairs)
            logger.info(f"Generated {len(qa_pairs)} QA pairs")
            return qa_pairs
        except Exception as e:
            ctx.state.errors.append(f"QA generation failed: {e}")
            logger.error(f"QA generation failed: {e}")
            raise

    # Step 4: QA Validation
    @g.step
    async def validate_qa_step(ctx: StepContext[ETLState, None, list[QAPair]]) -> list[QAPair]:
        """Validate and filter QA pairs."""
        logger.info("Validating QA pairs...")

        try:
            validated = await validate_qa_pairs(
                ctx.inputs,
                ctx.state.chunks,
                min_quality=0.6
            )
            ctx.state.validated_qa_pairs = validated
            ctx.state.stats["num_qa_validated"] = len(validated)
            ctx.state.stats["qa_pass_rate"] = len(validated) / len(ctx.inputs) if ctx.inputs else 0
            logger.info(f"Validated {len(validated)}/{len(ctx.inputs)} QA pairs")
            return validated
        except Exception as e:
            ctx.state.errors.append(f"QA validation failed: {e}")
            logger.error(f"QA validation failed: {e}")
            raise

    # Step 5: Build Fine-tuning Dataset
    @g.step
    async def build_ft_data_step(ctx: StepContext[ETLState, None, list[QAPair]]) -> ETLOutput:
        """Build fine-tuning dataset from validated QA pairs."""
        logger.info("Building fine-tuning dataset...")

        chunk_by_id = {c.id: c for c in ctx.state.chunks}

        ft_samples = []
        for qa in ctx.inputs:
            chunk = chunk_by_id.get(qa.chunk_id)
            if chunk:
                ft_samples.append(FineTuneSample(
                    instruction=qa.question,
                    input=f"Context:\n{chunk.text}",
                    output=qa.answer,
                    metadata={
                        "chunk_id": qa.chunk_id,
                        "source": chunk.source_path,
                        "quality_score": qa.quality_score,
                        "section": chunk.section_title,
                        "tags": chunk.tags,
                    }
                ))

        ctx.state.ft_samples = ft_samples
        ctx.state.stats["num_ft_samples"] = len(ft_samples)

        logger.info(f"Created {len(ft_samples)} fine-tuning samples")

        return ETLOutput(
            chunks=ctx.state.chunks,
            summaries=ctx.state.summaries,
            qa_pairs=ctx.state.validated_qa_pairs,
            ft_samples=ft_samples,
            stats=ctx.state.stats,
        )

    # Wire up the graph
    g.add(
        # Start -> Chunking
        g.edge_from(g.start_node).to(chunk_document_step),

        # Chunking -> Metadata
        g.edge_from(chunk_document_step).to(extract_metadata_step),

        # Metadata -> Summaries
        g.edge_from(extract_metadata_step).to(generate_summaries_step),

        # Summaries -> QA Generation
        g.edge_from(generate_summaries_step).to(generate_qa_step),

        # QA Generation -> Validation
        g.edge_from(generate_qa_step).to(validate_qa_step),

        # Validation -> Build FT Data
        g.edge_from(validate_qa_step).to(build_ft_data_step),

        # Build FT Data -> End
        g.edge_from(build_ft_data_step).to(g.end_node),
    )

    return g.build()


# Pre-built graph instance
_etl_graph = None


def get_etl_graph():
    """Get or create the ETL pipeline graph."""
    global _etl_graph
    if _etl_graph is None:
        _etl_graph = build_etl_graph()
    return _etl_graph


async def run_etl_pipeline(
    source_path: str,
    raw_text: str,
) -> ETLOutput:
    """
    Run the complete ETL pipeline on a document.
    
    Args:
        source_path: Path to source document
        raw_text: Raw text content
        
    Returns:
        ETLOutput with chunks, summaries, QA pairs, and FT samples
    """
    graph = get_etl_graph()
    state = ETLState(source_path=source_path, raw_text=raw_text)

    result = await graph.run(state=state)

    return result


def run_etl_pipeline_sync(
    source_path: str,
    raw_text: str,
) -> ETLOutput:
    """Synchronous version of run_etl_pipeline."""
    import asyncio

    try:
        # Check if there's already a running event loop
        loop = asyncio.get_running_loop()
        # If we're already in an async context, create a task
        # This requires nest_asyncio to work properly
        try:
            import nest_asyncio
            nest_asyncio.apply()
        except ImportError:
            raise RuntimeError(
                "nest_asyncio is required when calling from async context. "
                "Install with: uv add nest-asyncio"
            )
        return asyncio.run(run_etl_pipeline(source_path, raw_text))
    except RuntimeError:
        # No running loop, safe to use asyncio.run()
        return asyncio.run(run_etl_pipeline(source_path, raw_text))


if __name__ == "__main__":
    # Generate Mermaid diagram
    graph = build_etl_graph()
    print(graph.render(title="ETL Pipeline", direction="TB"))
