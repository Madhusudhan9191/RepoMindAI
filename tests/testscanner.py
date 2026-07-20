from backend.app.core.container import get_container
from backend.app.ingestion.ingest_pipeline import IngestionPipeline


def main():
    """
    Test the complete RAG retrieval pipeline.
    """

    container = get_container()

    try:
        # --------------------------------------------------
        # Step 1 : Index Repository
        # --------------------------------------------------
        pipeline = IngestionPipeline(".")

        pipeline.ingest()

        # --------------------------------------------------
        # Step 2 : Collection Statistics
        # --------------------------------------------------
        print("\n" + "=" * 80)
        print("QDRANT COLLECTION")
        print("=" * 80)

        info = container.qdrant_service.get_collection_info()

        print(f"Collection Name : {container.qdrant_service.COLLECTION_NAME}")
        print(f"Vector Count    : {info.points_count}")

        # --------------------------------------------------
        # Step 3 : Ask Question
        # --------------------------------------------------
        query = "How are embeddings generated?"

        print("\n" + "=" * 80)
        print(f"QUESTION : {query}")
        print("=" * 80)

        results = container.retriever.retrieve(query)

        if not results:
            print("No results found.")
            return

        # --------------------------------------------------
        # Step 4 : Display Results
        # --------------------------------------------------
        for index, result in enumerate(results, start=1):

            payload = result.payload

            print("\n" + "-" * 80)
            print(f"Result #{index}")
            print("-" * 80)

            print(f"Similarity : {result.score:.4f}")
            print(f"File       : {payload['path']}")
            print(f"Function   : {payload['name']}")
            print(f"Type       : {payload['type']}")
            print(
                f"Lines      : "
                f"{payload['start_line']} - {payload['end_line']}"
            )

            print("\nCode")
            print("-" * 80)
            print(payload["content"])

    finally:
        # Always release the Qdrant lock
        container.close()


if __name__ == "__main__":
    main()