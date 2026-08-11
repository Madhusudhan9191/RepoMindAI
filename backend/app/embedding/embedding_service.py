from collections import OrderedDict
import threading
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """
    Service responsible for generating embeddings for code chunks
    and documents using a SentenceTransformer model, with an LRU cache for query text.
    """

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    CACHE_SIZE = 512

    def __init__(self):
        self.model = SentenceTransformer(self.MODEL_NAME)
        self._query_cache = OrderedDict()
        self._cache_lock = threading.Lock()
        self.cache_hits = 0
        self.cache_misses = 0

    def embed_text(self, text: str) -> list[float]:
        """
        Generate an embedding for a single text.

        Args:
            text: Input text or code snippet.

        Returns:
            A list of floating-point values representing the embedding.
        """
        return self.model.encode(text).tolist()

    def embed_query(self, query: str) -> list[float]:
        """
        Generate or retrieve a cached embedding vector for a user query.
        """
        query_key = query.strip()
        with self._cache_lock:
            if query_key in self._query_cache:
                self.cache_hits += 1
                self._query_cache.move_to_end(query_key)
                return self._query_cache[query_key]

        vector = self.embed_text(query_key)

        with self._cache_lock:
            self.cache_misses += 1
            self._query_cache[query_key] = vector
            if len(self._query_cache) > self.CACHE_SIZE:
                self._query_cache.popitem(last=False)

        return vector

    def get_cache_stats(self) -> dict:
        return {
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "cached_queries": len(self._query_cache)
        }

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple documents in a single batched inference call.

        Args:
            documents: List of texts or code snippets.

        Returns:
            List of embedding vectors.
        """
        if not documents:
            return []
        return self.model.encode(documents, batch_size=32).tolist()

    def embed_chunk(self, chunk: dict) -> dict:
        """
        Generate an embedding for a code chunk and
        attach it to the chunk dictionary.

        Args:
            chunk: Dictionary produced by the CodeChunker.

        Returns:
            Updated chunk containing the embedding.
        """
        chunk["embedding"] = self.embed_text(chunk["content"])
        return chunk

    def embed_chunks(self, chunks: list[dict]) -> list[dict]:
        """
        Generate embeddings for multiple code chunks in batch.

        Args:
            chunks: List of chunk dictionaries.

        Returns:
            List of updated chunks containing embeddings.
        """
        if not chunks:
            return []

        texts = [chunk["content"] for chunk in chunks]
        embeddings = self.embed_documents(texts)

        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding

        return chunks

    def get_embedding_dimension(self) -> int:
        """
        Return the embedding dimension of the loaded model.
        """
        return self.model.get_embedding_dimension()

    def get_model_name(self) -> str:
        """
        Return the name of the embedding model.
        """
        return self.MODEL_NAME