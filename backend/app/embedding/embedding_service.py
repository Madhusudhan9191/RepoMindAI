from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """
    Service responsible for generating embeddings for code chunks
    and documents using a SentenceTransformer model.
    """

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self):
        self.model = SentenceTransformer(self.MODEL_NAME)

    def embed_text(self, text: str) -> list[float]:
        """
        Generate an embedding for a single text.

        Args:
            text: Input text or code snippet.

        Returns:
            A list of floating-point values representing the embedding.
        """
        return self.model.encode(text).tolist()

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple documents.

        Args:
            documents: List of texts or code snippets.

        Returns:
            List of embedding vectors.
        """
        return self.model.encode(documents).tolist()

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
        Generate embeddings for multiple code chunks.

        Args:
            chunks: List of chunk dictionaries.

        Returns:
            List of updated chunks containing embeddings.
        """
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