import os

# Set environment variables for testing before importing anything that initializes the Container
os.environ["LLM_PROVIDER"] = "mock"
os.environ["LLM_MODEL"] = "mock-gpt-model"

from backend.app.core.container import get_container
from backend.app.ingestion.ingest_pipeline import IngestionPipeline


def main():
    """
    Test the end-to-end RAG generation pipeline.
    """
    print("=" * 80)
    print("RUNNING END-TO-END RAG PIPELINE TEST")
    print("=" * 80)

    # 1. Initialize container
    container = get_container()

    try:
        # 2. Index the current repository to load latest files into Qdrant
        print("\nStep 1: Indexing current repository...")
        pipeline = IngestionPipeline(".")
        pipeline.ingest()

        # 3. Formulate RAG query
        query = "How do we generate embeddings for a code chunk?"
        print(f"\nStep 2: Submitting query: '{query}'")

        # 4. Perform vector search and context assembly
        print("\nStep 3: Retrieving context from Qdrant vector database...")
        results = container.retriever.retrieve(query, top_k=3)
        context_string = container.retriever.get_context(query, top_k=3)

        # 5. Format prompt message package
        print("\nStep 4: Building RAG prompt message package...")
        messages = container.prompt_builder.build_rag_prompt(query, context_string)

        print("-" * 80)
        print("COMPOSED RAG SYSTEM INSTRUCTION:")
        print(messages[0]["content"])
        print("-" * 80)
        print("COMPOSED USER AND CONTEXT MESSAGE:")
        print(messages[1]["content"][:300] + "...\n[Context Truncated for View]...")
        print("-" * 80)

        # 6. Execute LLM generation
        print("\nStep 5: Invoking LLM generation service...")
        response = container.llm_service.generate(messages)

        print("\n" + "=" * 80)
        print("LLM GENERATED RESPONSE:")
        print("=" * 80)
        print(response)

        # 7. Print citations
        print("\n" + "=" * 80)
        print("SOURCE CITATIONS:")
        print("=" * 80)
        for idx, res in enumerate(results, 1):
            p = res.payload
            print(
                f"{idx}. File: {p['path']}\n"
                f"   Function: {p['name']} (Lines {p['start_line']} - {p['end_line']})\n"
                f"   Search Score: {res.score:.4f}\n"
            )

    finally:
        # Always release database lock
        container.close()


if __name__ == "__main__":
    main()
