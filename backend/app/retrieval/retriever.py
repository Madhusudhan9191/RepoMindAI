from typing import List

from backend.app.embedding.embedding_service import EmbeddingService
from backend.app.vectorstore.qdrant_service import QdrantService
from backend.app.retrieval.mmr import maximum_marginal_relevance
from backend.app.core.telemetry import TelemetryCollector


class RetrievalConfig:
    """
    Configuration parameters for search, retrieval, and diversification.
    Provides standard parameter defaults for the entire retrieval pipeline.
    """

    def __init__(
        self,
        top_k: int = 5,
        fetch_k: int = 20,
        rewrite_query: bool = True,
        use_mmr: bool = True,
        lambda_mult: float = 0.5,
        use_reranker: bool = True,
        reranker_top_k: int = 10,
        use_compressor: bool = True,
    ):
        self.top_k = top_k
        self.fetch_k = fetch_k
        self.rewrite_query = rewrite_query
        self.use_mmr = use_mmr
        self.lambda_mult = lambda_mult
        self.use_reranker = use_reranker
        self.reranker_top_k = reranker_top_k
        self.use_compressor = use_compressor


class Retriever:
    """
    Retrieves relevant code chunks from the vector database.
    Supports standard semantic search, diversified MMR search, Cross-Encoder reranking,
    and context compression.
    """

    DEFAULT_TOP_K = 5

    def __init__(
        self,
        embedding_service: EmbeddingService,
        qdrant_service: QdrantService,
        reranker_service = None,
        compressor = None,
        bm25_retriever = None,
    ):
        self.embedding_service = embedding_service
        self.qdrant_service = qdrant_service
        self.reranker_service = reranker_service
        self.compressor = compressor
        if bm25_retriever is None:
            from backend.app.retrieval.bm25_retriever import BM25Retriever
            self.bm25_retriever = BM25Retriever()
        else:
            self.bm25_retriever = bm25_retriever

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        use_mmr: bool = True,
        lambda_mult: float = 0.5,
        fetch_k: int = 20,
        use_reranker: bool = True,
        reranker_top_k: int = 10,
        config: RetrievalConfig = None,
        telemetry: TelemetryCollector = None,
    ):
        """
        Retrieves the most relevant code chunks through a multi-stage pipeline:
        1. Retrieval: Fetch candidate points from Qdrant.
        2. Diversification: Filter candidates using MMR (if enabled).
        3. Reranking: Re-score candidates using a Cross-Encoder (if enabled).

        Args:
            query:           Natural language query.
            top_k:           Number of final chunks to retrieve.
            use_mmr:         If True, run Maximum Marginal Relevance diversification.
            lambda_mult:     Coefficient determining diversity weight (0.0 to 1.0).
            fetch_k:         Number of candidate chunks to fetch initially.
            use_reranker:    If True, run Cross-Encoder reranking.
            reranker_top_k:  Number of candidates to evaluate with the reranker.
            config:          Optional RetrievalConfig object override.
            telemetry:       Optional TelemetryCollector collector.

        Returns:
            List of Qdrant ScoredPoint objects.
        """
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        # Resolve config overrides
        if config is not None:
            top_k = config.top_k
            fetch_k = config.fetch_k
            use_mmr = config.use_mmr
            lambda_mult = config.lambda_mult
            use_reranker = config.use_reranker
            reranker_top_k = config.reranker_top_k

        if telemetry:
            telemetry.set_config(
                use_mmr=use_mmr,
                use_reranker=use_reranker,
                use_compressor=config.use_compressor if config else True,
            )

        # --- Stage 1: Candidate Ingestion (Dense Qdrant Search + BM25 Lexical Search) ---
        need_vectors = use_mmr
        
        if use_mmr:
            db_limit = fetch_k
        elif use_reranker:
            db_limit = reranker_top_k
        else:
            db_limit = top_k

        if telemetry:
            telemetry.start_stage("vector_search")

        # 1A. Qdrant Dense Search
        candidates = self.qdrant_service.search(
            query=query,
            limit=db_limit,
            with_vectors=need_vectors,
        )

        # 1B. BM25 Lexical Search (Candidate Union)
        if hasattr(self, "bm25_retriever") and self.bm25_retriever and candidates:
            try:
                # Ensure BM25 index contains current candidates if not indexed
                if self.bm25_retriever.doc_count == 0:
                    self.bm25_retriever.index_chunks([c.payload for c in candidates if c.payload])

                bm25_results = dict(self.bm25_retriever.search(query, top_k=db_limit))
                for c in candidates:
                    cid = c.payload.get("id") if c.payload else None
                    if cid in bm25_results:
                        c.payload["bm25_score"] = float(bm25_results[cid])
                    else:
                        c.payload["bm25_score"] = 0.0
            except Exception:
                pass

        if telemetry:
            telemetry.end_stage()
            telemetry.set_count("candidates_retrieved", len(candidates))
            if candidates:
                avg_sim = sum(c.score for c in candidates) / len(candidates)
                telemetry.set_scores(avg_vector_similarity=round(avg_sim, 4))

        if not candidates:
            return []

        # --- Stage 2: MMR Diversification ---
        if use_mmr:
            query_vector = self.embedding_service.embed_text(query)
            mmr_limit = reranker_top_k if use_reranker else top_k
            
            if telemetry:
                telemetry.start_stage("mmr")
            candidates = maximum_marginal_relevance(
                query_vector=query_vector,
                candidate_points=candidates,
                top_k=mmr_limit,
                lambda_mult=lambda_mult,
            )
            if telemetry:
                telemetry.end_stage()
                telemetry.set_count("after_mmr", len(candidates))
                mmr_scores = [
                    c.payload.get("mmr_score")
                    for c in candidates
                    if c.payload and "mmr_score" in c.payload
                ]
                if mmr_scores:
                    avg_mmr = sum(mmr_scores) / len(mmr_scores)
                    telemetry.set_scores(avg_mmr_score=round(avg_mmr, 4))

        # --- Stage 3: Cross-Encoder Reranking ---
        if use_reranker and self.reranker_service:
            if telemetry:
                telemetry.start_stage("rerank")
            candidates = self.reranker_service.rerank(query, candidates)
            if telemetry:
                telemetry.end_stage()
                telemetry.set_count("after_reranker", len(candidates))
                rerank_scores = [
                    c.payload.get("rerank_score")
                    for c in candidates
                    if c.payload and "rerank_score" in c.payload
                ]
                if rerank_scores:
                    avg_rerank = sum(rerank_scores) / len(rerank_scores)
                    telemetry.set_scores(avg_rerank_score=round(avg_rerank, 4))

        # Slice to the final requested count
        return candidates[:top_k]

    def retrieve_payloads(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        config: RetrievalConfig = None,
        telemetry: TelemetryCollector = None,
    ) -> List[dict]:
        """
        Retrieve only payloads.

        Useful for Prompt Builder.
        """
        results = self.retrieve(
            query=query, top_k=top_k, config=config, telemetry=telemetry
        )

        return [
            result.payload
            for result in results
        ]

    def pretty_print(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        config: RetrievalConfig = None,
    ):
        """
        Print retrieved chunks in a readable format.
        """
        results = self.retrieve(query=query, top_k=top_k, config=config)

        if not results:
            print("\nNo matching chunks found.")
            return

        print("\n" + "=" * 80)
        print(f"Query : {query}")
        print("=" * 80)

        for index, result in enumerate(results, start=1):
            payload = result.payload
            mmr_str = f" | MMR Score: {payload['mmr_score']:.4f}" if payload and "mmr_score" in payload else ""
            rerank_str = f" | Rerank Score: {payload['rerank_score']:.4f}" if payload and "rerank_score" in payload else ""

            print(f"\nResult #{index}")
            print("-" * 80)
            print(f"Score      : {result.score:.4f}{mmr_str}{rerank_str}")
            print(f"File       : {payload['path']}")
            print(f"Function   : {payload['name']}")
            print(f"Type       : {payload['type']}")
            print(
                f"Lines      : "
                f"{payload['start_line']} - {payload['end_line']}"
            )

            print("\nContent")
            print("-" * 80)
            print(payload["content"])

    def get_context(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        config: RetrievalConfig = None,
        telemetry: TelemetryCollector = None,
    ) -> str:
        """
        Build context string from retrieved chunks.
        Applies line-level semantic context compression if enabled.
        """
        payloads = self.retrieve_payloads(
            query=query, top_k=top_k, config=config, telemetry=telemetry
        )

        # Determine if compressor should run
        use_compress = True
        if config is not None:
            use_compress = config.use_compressor

        context = []

        for payload in payloads:
            code_content = payload.get("content", "")
            
            orig_lines = len(code_content.splitlines())
            orig_tokens = len(code_content) // 4
            
            if telemetry:
                current_orig_lines = telemetry.counts["original_context_lines"]
                current_orig_tokens = telemetry.counts["original_context_tokens"]
                telemetry.set_count("original_context_lines", current_orig_lines + orig_lines)
                telemetry.set_count("original_context_tokens", current_orig_tokens + orig_tokens)

            # Apply Context Compressor
            if use_compress and self.compressor:
                if telemetry:
                    telemetry.start_stage("compression")
                code_content = self.compressor.compress(
                    query=query,
                    code_content=code_content,
                    filepath=payload.get("path", "")
                )
                if telemetry:
                    telemetry.end_stage()

            comp_lines = len(code_content.splitlines())
            comp_tokens = len(code_content) // 4
            
            if telemetry:
                current_comp_lines = telemetry.counts["compressed_context_lines"]
                current_comp_tokens = telemetry.counts["compressed_context_tokens"]
                telemetry.set_count("compressed_context_lines", current_comp_lines + comp_lines)
                telemetry.set_count("compressed_context_tokens", current_comp_tokens + comp_tokens)

            context.append(
                f"""File: {payload['path']}
Function: {payload['name']}
Lines: {payload['start_line']} - {payload['end_line']}
Code:
{code_content}"""
            )

        return "\n\n".join(context)