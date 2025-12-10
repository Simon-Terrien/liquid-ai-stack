# liquid_shared/vectordb.py
"""
Vector database wrapper using ChromaDB.

Provides a simple interface for storing and querying document embeddings.
Supports both in-memory and persistent storage.
"""
import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from .config import EMBEDDING_DIM

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Chroma-based vector store for document chunks.
    
    Example:
        ```python
        store = VectorStore("/data/vectordb")
        store.add(
            ids=["doc1_chunk1"],
            embeddings=[[0.1, 0.2, ...]],
            metadatas=[{"source": "doc1.pdf"}],
            documents=["The content..."]
        )
        results = store.query(query_embedding=[0.1, 0.2, ...], top_k=5)
        ```
    """

    def __init__(
        self,
        persist_dir: Path | None = None,
        collection_name: str = "documents",
        distance_metric: str = "cosine"
    ):
        """
        Initialize the vector store.
        
        Args:
            persist_dir: Directory for persistent storage. If None, uses in-memory.
            collection_name: Name of the collection
            distance_metric: Distance metric ("cosine", "l2", "ip")
        """
        self.persist_dir = persist_dir
        self.collection_name = collection_name

        # Configure client
        if persist_dir:
            persist_dir = Path(persist_dir)
            persist_dir.mkdir(parents=True, exist_ok=True)

            self.client = chromadb.Client(Settings(
                persist_directory=str(persist_dir),
                anonymized_telemetry=False,
                is_persistent=True,
            ))
            logger.info(f"Initialized persistent vector store at {persist_dir}")
        else:
            self.client = chromadb.Client(Settings(
                anonymized_telemetry=False,
            ))
            logger.info("Initialized in-memory vector store")

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": distance_metric},
        )

        logger.info(f"Collection '{collection_name}' has {self.collection.count()} documents")

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
        documents: list[str],
    ) -> None:
        """
        Add documents to the vector store.
        
        Args:
            ids: Unique identifiers for each document
            embeddings: Vector embeddings (list of float lists)
            metadatas: Metadata dicts for each document
            documents: Raw text content
        """
        # Validate inputs
        n = len(ids)
        assert len(embeddings) == n, f"embeddings count mismatch: {len(embeddings)} vs {n}"
        assert len(metadatas) == n, f"metadatas count mismatch: {len(metadatas)} vs {n}"
        assert len(documents) == n, f"documents count mismatch: {len(documents)} vs {n}"

        # ChromaDB doesn't like None values in metadata
        clean_metadatas = []
        for meta in metadatas:
            clean_meta = {}
            for k, v in meta.items():
                if v is not None:
                    # Convert lists to strings for ChromaDB compatibility
                    if isinstance(v, list):
                        clean_meta[k] = ",".join(str(x) for x in v)
                    else:
                        clean_meta[k] = v
            clean_metadatas.append(clean_meta)

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=clean_metadatas,
            documents=documents,
        )

        logger.info(f"Added {n} documents to collection")

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Query the vector store for similar documents.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            where: Metadata filter (e.g., {"source": "doc1.pdf"})
            where_document: Document content filter
            
        Returns:
            Dict with keys: ids, embeddings, metadatas, documents, distances
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            where_document=where_document,
        )

        return results

    def get(
        self,
        ids: list[str] | None = None,
        where: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """
        Get documents by ID or filter.
        
        Args:
            ids: Specific document IDs to retrieve
            where: Metadata filter
            limit: Maximum number of results
            
        Returns:
            Dict with keys: ids, embeddings, metadatas, documents
        """
        return self.collection.get(
            ids=ids,
            where=where,
            limit=limit,
        )

    def update(
        self,
        ids: list[str],
        embeddings: list[list[float]] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
        documents: list[str] | None = None,
    ) -> None:
        """Update existing documents."""
        self.collection.update(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )

    def delete(
        self,
        ids: list[str] | None = None,
        where: dict[str, Any] | None = None,
    ) -> None:
        """Delete documents by ID or filter."""
        self.collection.delete(ids=ids, where=where)

    def count(self) -> int:
        """Get total number of documents in collection."""
        return self.collection.count()

    def reset(self) -> None:
        """Delete all documents in the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.warning(f"Reset collection '{self.collection_name}'")

    def persist(self) -> None:
        """Persist vector store to disk (if using persistent storage)."""
        # ChromaDB 0.4+ auto-persists, no need to call persist()
        # Keeping this method for backwards compatibility
        pass


class EmbeddingService:
    """
    Wrapper for generating embeddings using sentence-transformers.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        """
        Initialize the embedding service.
        
        Args:
            model_name: HuggingFace model ID for embeddings
        """
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

        logger.info(f"Loaded embedding model {model_name} (dim={self.dimension})")

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> list[list[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
            
        Returns:
            List of embedding vectors
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    def encode_single(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        return self.encode([text])[0]


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)

    print("Testing VectorStore...")

    # Test in-memory store
    store = VectorStore(collection_name="test")

    # Add some test data
    store.add(
        ids=["1", "2", "3"],
        embeddings=[
            [0.1] * EMBEDDING_DIM,
            [0.2] * EMBEDDING_DIM,
            [0.3] * EMBEDDING_DIM,
        ],
        metadatas=[
            {"source": "doc1.txt"},
            {"source": "doc2.txt"},
            {"source": "doc3.txt"},
        ],
        documents=[
            "First document about AI.",
            "Second document about ML.",
            "Third document about NLP.",
        ],
    )

    print(f"Store has {store.count()} documents")

    # Query
    results = store.query([0.15] * EMBEDDING_DIM, top_k=2)
    print(f"Query results: {results['ids']}")

    # Cleanup
    store.reset()
    print("Test completed!")
