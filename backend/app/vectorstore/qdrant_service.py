import uuid
from typing import List

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)


class QdrantService:
    """
    Handles all interactions with the Qdrant vector database.

    Responsibilities:
    - Create collection
    - Insert vectors
    - Search vectors
    - Collection management
    """

    COLLECTION_NAME = "repo_chunks"

    def __init__(
        self,
        embedding_service,
        db_path: str = "./qdrant_data",
    ):
        self.embedding_service = embedding_service

        self.client = QdrantClient(path=db_path)

        self.vector_size = (
            self.embedding_service.get_embedding_dimension()
        )

    # --------------------------------------------------
    # Collection Management
    # --------------------------------------------------

    def create_collection(self):
        """
        Create the collection if it doesn't already exist.
        """

        collections = self.client.get_collections().collections

        existing = {
            collection.name
            for collection in collections
        }

        if self.COLLECTION_NAME in existing:
            return

        self.client.create_collection(
            collection_name=self.COLLECTION_NAME,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=Distance.COSINE,
            ),
        )

    def delete_collection(self):
        """
        Delete collection.
        """

        self.client.delete_collection(
            collection_name=self.COLLECTION_NAME
        )

    def get_collection_info(self):
        """
        Return collection metadata.
        """

        return self.client.get_collection(
            self.COLLECTION_NAME
        )

    # --------------------------------------------------
    # Insert Operations
    # --------------------------------------------------

    def _generate_id(
        self,
        chunk_id: str,
    ) -> str:
        """
        Generate deterministic UUID.
        """

        return str(
            uuid.uuid5(
                uuid.NAMESPACE_DNS,
                chunk_id,
            )
        )

    def insert_chunk(
        self,
        chunk: dict,
    ):
        """
        Insert one chunk.
        """

        self.insert_chunks([chunk])

    def insert_chunks(
        self,
        chunks: List[dict],
    ):
        """
        Batch insert chunks.
        """

        if not chunks:
            return

        points = []

        for chunk in chunks:

            points.append(
                PointStruct(
                    id=self._generate_id(
                        chunk["id"]
                    ),
                    vector=chunk["embedding"],
                    payload={
                        "id": chunk["id"],
                        "type": chunk["type"],
                        "name": chunk["name"],
                        "path": chunk["path"],
                        "language": chunk["language"],
                        "start_line": chunk["start_line"],
                        "end_line": chunk["end_line"],
                        "content": chunk["content"],
                    },
                )
            )

        self.client.upsert(
            collection_name=self.COLLECTION_NAME,
            points=points,
        )

    # --------------------------------------------------
    # Search
    # --------------------------------------------------

    def search(
        self,
        query: str,
        limit: int = 5,
        with_vectors: bool = False,
    ):
        """
        Semantic search.
        """

        query_vector = (
            self.embedding_service.embed_text(
                query
            )
        )

        response = self.client.query_points(
            collection_name=self.COLLECTION_NAME,
            query=query_vector,
            limit=limit,
            with_vectors=with_vectors,
        )

        return response.points

    # --------------------------------------------------
    # Utility
    # --------------------------------------------------

    def count(self) -> int:
        """
        Number of indexed vectors.
        """

        info = self.get_collection_info()

        return info.points_count

    def delete_chunks_by_path(self, path: str):
        """
        Delete all points matching a specific file path (used for incremental ingestion/deletion sync).
        """
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            self.client.delete(
                collection_name=self.COLLECTION_NAME,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="path",
                            match=MatchValue(value=path)
                        )
                    ]
                )
            )
        except Exception as e:
            pass

    def clear(self):
        """
        Delete all points in the collection. Fallback to recreating the collection
        if needed.
        """
        try:
            from qdrant_client.models import Filter
            self.client.delete(
                collection_name=self.COLLECTION_NAME,
                points_selector=Filter()
            )
        except Exception:
            try:
                self.delete_collection()
            except Exception:
                pass
            self.create_collection()

    def close(self):
        """
        Release database lock.
        """

        try:
            self.client.close()
        except Exception:
            pass