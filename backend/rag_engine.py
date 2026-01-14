import numpy as np
from typing import List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity
import ollama
from concurrent.futures import ThreadPoolExecutor

class RAGEngine:
    """
    Optimized Retrieval-Augmented Generation Engine with:
    - Document chunking (overlapping, small chunks)
    - Ollama embedding generation (parallel + caching)
    - Efficient top-k retrieval
    """

    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim
        self.chunks: List[Dict[str, Any]] = []  # {"doc_id", "chunk", "chunk_index"}
        self.embeddings = np.empty((0, embedding_dim), dtype=np.float32)
        self.embedding_cache: Dict[int, np.ndarray] = {}  # doc_id -> embeddings
        print(f"✅ RAG Engine initialized (embedding_dim={embedding_dim})")

    # ------------------ Document Handling ------------------
    def add_document(
        self, doc_id: int, text: str, chunk_size: int = 400, chunk_overlap: int = 50
    ) -> int:
        """
        Add a document: split into chunks, generate embeddings, store with caching.

        Returns:
            int: number of chunks created
        """
        text = text.strip()
        if not text:
            return 0

        new_chunks = self.create_chunks(text, chunk_size, chunk_overlap)
        if not new_chunks:
            return 0

        # Generate or fetch embeddings from cache
        if doc_id in self.embedding_cache:
            new_embeddings = self.embedding_cache[doc_id]
        else:
            new_embeddings = self.generate_embeddings(new_chunks)
            self.embedding_cache[doc_id] = new_embeddings

        # Store chunks with chunk_index
        for i, chunk in enumerate(new_chunks):
            self.chunks.append({
                "doc_id": doc_id,
                "chunk": chunk,
                "chunk_index": i
            })

        # Stack embeddings
        if len(self.embeddings) == 0:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

        return len(new_chunks)

    # ------------------ Chunking ------------------
    def create_chunks(self, text: str, chunk_size: int = 400, chunk_overlap: int = 50) -> List[str]:
        """
        Split text into overlapping word chunks.
        """
        words = text.split()
        if len(words) <= chunk_size:
            return [" ".join(words)]

        chunks = []
        step = chunk_size - chunk_overlap
        for i in range(0, len(words), step):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
            if i + chunk_size >= len(words):
                break
        return chunks

    # ------------------ Embedding Generation ------------------
    def generate_embeddings(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        """
        Generate embeddings using Ollama nomic-embed-text model.
        Parallelized for speed, fallback to random if failed.

        Args:
            texts: List of text chunks
            batch_size: Number of chunks to process in parallel

        Returns:
            np.ndarray: shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        def embed_chunk(chunk: str):
            truncated = chunk[:2000]  # Ollama safe limit
            try:
                resp = ollama.embeddings(model='nomic-embed-text', prompt=truncated)
                return np.array(resp['embedding'], dtype=np.float32)
            except Exception as e:
                print(f"⚠️ Ollama embedding failed: {e}, using random fallback")
                return np.random.rand(self.embedding_dim).astype(np.float32)

        # Run in parallel
        with ThreadPoolExecutor() as executor:
            embeddings = list(executor.map(embed_chunk, texts))

        return np.vstack(embeddings)

    # ------------------ Retrieval ------------------
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top_k most similar chunks to a query string.

        Args:
            query: raw query text
            top_k: number of chunks to return

        Returns:
            List of dicts: {"doc_id", "chunk", "score", "chunk_index"}
        """
        # Early return if no embeddings exist
        if len(self.chunks) == 0 or self.embeddings.shape[0] == 0:
            print(f"⚠️ No documents loaded. Returning empty results.")
            return []

        # Generate query embedding
        try:
            resp = ollama.embeddings(model='nomic-embed-text', prompt=query[:2000])
            query_embedding = np.array([resp['embedding']], dtype=np.float32)
        except Exception as e:
            print(f"⚠️ Query embedding failed: {e}, using random fallback")
            query_embedding = np.random.rand(1, self.embedding_dim).astype(np.float32)

        # Ensure embeddings are not empty
        if self.embeddings.shape[0] == 0:
            print(f"⚠️ Document embeddings are empty. Did embedding generation fail?")
            return []

        # Compute similarity
        sims = cosine_similarity(query_embedding, self.embeddings)[0]
        top_indices = sims.argsort()[-top_k:][::-1]

        return [
            {
                "doc_id": self.chunks[i]["doc_id"],
                "chunk": self.chunks[i]["chunk"],
                "score": float(sims[i]),
                "chunk_index": self.chunks[i]["chunk_index"]
            }
            for i in top_indices
        ]

    # ------------------ Stats ------------------
    def get_stats(self) -> Dict[str, Any]:
        """Return RAG engine stats."""
        return {
            "total_chunks": len(self.chunks),
            "embedding_dim": self.embedding_dim,
            "unique_documents": len(set(c["doc_id"] for c in self.chunks)),
            "embeddings_shape": self.embeddings.shape if len(self.embeddings) > 0 else "Empty"
        }
