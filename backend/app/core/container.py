"""
Dependency Injection Container

Creates and manages shared service instances for the entire application.

Only ONE instance of each service exists.

Architecture:

Container
│
├── EmbeddingService
├── QdrantService
└── Retriever
"""

import os
from dotenv import load_dotenv
from backend.app.embedding.embedding_service import EmbeddingService
from backend.app.vectorstore.qdrant_service import QdrantService
from backend.app.retrieval import Retriever, ContextBuilder
from backend.app.retrieval.reranker import RerankerService
from backend.app.retrieval.compressor import ContextCompressor
from backend.app.prompts.prompt_builder import PromptBuilder
from backend.app.services.query_rewriter import QueryRewriter
from backend.app.services.llm_service import get_llm_provider
from backend.app.services.rag_service import RAGService


class Container:
    """
    Application Dependency Container.
    """

    def __init__(self):

        print("Initializing RepoMindAI Container...")
        
        # Load environment configuration
        load_dotenv()

        # Load embedding model only once
        self.embedding_service = EmbeddingService()

        # Create one Qdrant client
        self.qdrant_service = QdrantService(
            self.embedding_service
        )

        # Ensure collection exists
        self.qdrant_service.create_collection()

        # Configurable Reranker Service
        self.reranker_service = RerankerService()

        # Shared Context Compressor
        self.context_compressor = ContextCompressor()

        # Shared BM25 Retriever
        from backend.app.retrieval.bm25_retriever import BM25Retriever
        self.bm25_retriever = BM25Retriever()

        # Shared Retriever
        self.retriever = Retriever(
            embedding_service=self.embedding_service,
            qdrant_service=self.qdrant_service,
            reranker_service=self.reranker_service,
            compressor=self.context_compressor,
            bm25_retriever=self.bm25_retriever,
        )

        # Shared Context Builder
        self.context_builder = ContextBuilder()

        # Shared Prompt Builder
        self.prompt_builder = PromptBuilder()

        # Configurable LLM Provider
        provider = os.getenv("LLM_PROVIDER", "mock")
        model = os.getenv("LLM_MODEL", "mock-model")
        api_key = os.getenv("LLM_API_KEY")
        api_base = os.getenv("LLM_API_BASE")

        self.llm_service = get_llm_provider(
            provider_name=provider,
            model_name=model,
            api_key=api_key,
            api_base=api_base
        )

        # Shared Query Rewriter
        self.query_rewriter = QueryRewriter(
            llm_service=self.llm_service
        )

        # Shared RAG Service Orchestrator
        self.rag_service = RAGService(
            retriever=self.retriever,
            context_builder=self.context_builder,
            prompt_builder=self.prompt_builder,
            llm_service=self.llm_service,
            query_rewriter=self.query_rewriter
        )

        print(f"Container initialized successfully. LLM Provider: {provider} ({model})")

    def configure_llm(self, provider: str, model: str, api_key: str = None, api_base: str = None):
        """
        Hot-swap the active LLM provider at runtime.
        """
        print(f"Re-configuring LLM Provider: {provider} ({model})...")
        self.llm_service = get_llm_provider(
            provider_name=provider,
            model_name=model,
            api_key=api_key,
            api_base=api_base
        )
        # Update active RAGService and QueryRewriter LLM references
        self.query_rewriter.llm_service = self.llm_service
        self.rag_service.llm_service = self.llm_service
        print("LLM Provider re-configured successfully.")

    def close(self):
        """
        Release all resources.
        """
        print("Closing RepoMindAI Container...")

        self.qdrant_service.close()

        print("Resources released.")


# ---------------------------------------------------
# Singleton Instance
# ---------------------------------------------------

_container = None


def get_container():

    global _container

    if _container is None:
        _container = Container()

    return _container