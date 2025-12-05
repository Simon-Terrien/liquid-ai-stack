# mcp_tools/servers/rag_server.py
"""
MCP Server for RAG tools.

Exposes retrieval and search capabilities via Model Context Protocol.
Compatible with Pydantic AI's MCP client integration.
"""
import logging

from liquid_shared import (
    DATA_DIR,
    EmbeddingService,
    VectorStore,
)
from mcp.server.fastmcp import Context, FastMCP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("Liquid RAG Tools")

# Paths
VDB_DIR = DATA_DIR / "vectordb"

# Services (lazy loaded)
_embedder: EmbeddingService | None = None
_vstore: VectorStore | None = None


def get_embedder() -> EmbeddingService:
    global _embedder
    if _embedder is None:
        _embedder = EmbeddingService()
    return _embedder


def get_vstore() -> VectorStore:
    global _vstore
    if _vstore is None:
        _vstore = VectorStore(VDB_DIR)
    return _vstore


# MCP Tools

@mcp.tool()
async def search_documents(
    ctx: Context,
    query: str,
    top_k: int = 5,
) -> str:
    """
    Search for relevant document chunks using semantic similarity.
    
    Args:
        query: The search query
        top_k: Number of results to return (default 5, max 20)
        
    Returns:
        Formatted search results with source information
    """
    logger.info(f"MCP search_documents: {query[:50]}...")

    embedder = get_embedder()
    vstore = get_vstore()

    # Clamp top_k
    top_k = max(1, min(20, top_k))

    # Generate query embedding
    query_embedding = embedder.encode_single(query)

    # Search
    results = vstore.query(query_embedding=query_embedding, top_k=top_k)

    # Format results
    if not results.get("ids") or not results["ids"][0]:
        return "No relevant documents found."

    formatted = []
    ids = results["ids"][0]
    documents = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    for i, doc_id in enumerate(ids):
        score = 1 - distances[i] if i < len(distances) else 0
        meta = metadatas[i] if i < len(metadatas) else {}
        text = documents[i] if i < len(documents) else ""

        source = meta.get("source", "unknown")
        section = meta.get("section_title", "")

        header = f"[Result {i+1}] Source: {source}"
        if section:
            header += f" | Section: {section}"
        header += f" | Score: {score:.2f}"

        formatted.append(f"{header}\n{text}")

    return "\n\n---\n\n".join(formatted)


@mcp.tool()
async def get_document_chunk(
    ctx: Context,
    chunk_id: str,
) -> str:
    """
    Retrieve a specific document chunk by ID.
    
    Args:
        chunk_id: The unique identifier of the chunk
        
    Returns:
        The chunk text and metadata, or error message if not found
    """
    logger.info(f"MCP get_document_chunk: {chunk_id}")

    vstore = get_vstore()

    results = vstore.get(ids=[chunk_id])

    if not results.get("ids") or not results["ids"]:
        return f"Chunk not found: {chunk_id}"

    text = results.get("documents", [""])[0]
    meta = results.get("metadatas", [{}])[0]

    output = f"Chunk ID: {chunk_id}\n"
    output += f"Source: {meta.get('source', 'unknown')}\n"
    if meta.get("section_title"):
        output += f"Section: {meta['section_title']}\n"
    output += f"\n{text}"

    return output


@mcp.tool()
async def search_by_source(
    ctx: Context,
    source_path: str,
    limit: int = 10,
) -> str:
    """
    Find all chunks from a specific source document.
    
    Args:
        source_path: Path or partial path of the source document
        limit: Maximum number of chunks to return
        
    Returns:
        List of chunks from the specified source
    """
    logger.info(f"MCP search_by_source: {source_path}")

    vstore = get_vstore()

    # Note: ChromaDB's where clause requires exact match
    # For partial matching, we'd need to implement differently
    results = vstore.get(
        where={"source": {"$contains": source_path}},
        limit=min(limit, 50),
    )

    if not results.get("ids"):
        return f"No chunks found from source: {source_path}"

    formatted = []
    for i, doc_id in enumerate(results["ids"]):
        text = results.get("documents", [])[i] if i < len(results.get("documents", [])) else ""
        meta = results.get("metadatas", [])[i] if i < len(results.get("metadatas", [])) else {}

        section = meta.get("section_title", f"Chunk {i+1}")
        formatted.append(f"[{section}]\n{text[:500]}...")

    return "\n\n---\n\n".join(formatted)


@mcp.tool()
async def get_collection_info(ctx: Context) -> str:
    """
    Get information about the document collection.
    
    Returns:
        Statistics about the indexed documents
    """
    vstore = get_vstore()

    count = vstore.count()

    return f"""Document Collection Info:
- Total chunks indexed: {count}
- Collection name: {vstore.collection_name}
- Storage: {vstore.persist_dir or 'in-memory'}"""


@mcp.tool()
async def search_by_tags(
    ctx: Context,
    tags: str,
    top_k: int = 5,
) -> str:
    """
    Search for chunks with specific tags.
    
    Args:
        tags: Comma-separated list of tags to search for
        top_k: Number of results to return
        
    Returns:
        Chunks matching the specified tags
    """
    logger.info(f"MCP search_by_tags: {tags}")

    vstore = get_vstore()

    # Parse tags
    tag_list = [t.strip().lower() for t in tags.split(",")]

    # Search by tags in metadata
    # Note: This is a simplified implementation
    all_results = vstore.get(limit=100)

    matching = []
    if all_results.get("ids"):
        for i, doc_id in enumerate(all_results["ids"]):
            meta = all_results.get("metadatas", [])[i] if i < len(all_results.get("metadatas", [])) else {}
            chunk_tags = meta.get("tags", "").lower()

            if any(tag in chunk_tags for tag in tag_list):
                text = all_results.get("documents", [])[i] if i < len(all_results.get("documents", [])) else ""
                matching.append({
                    "id": doc_id,
                    "text": text,
                    "meta": meta,
                })

    if not matching:
        return f"No chunks found with tags: {tags}"

    # Format results
    formatted = []
    for m in matching[:top_k]:
        source = m["meta"].get("source", "unknown")
        section = m["meta"].get("section_title", "")

        header = f"Source: {source}"
        if section:
            header += f" | {section}"

        formatted.append(f"[{header}]\n{m['text'][:500]}...")

    return "\n\n---\n\n".join(formatted)


# Server runner
def main():
    """Run the MCP server."""
    import argparse

    parser = argparse.ArgumentParser(description="Run RAG MCP server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="Transport protocol"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Port for SSE transport"
    )

    args = parser.parse_args()

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport="sse", port=args.port)


if __name__ == "__main__":
    main()
