import logging
import time
import json
import uuid
import copy
from backend.app.retrieval.context_builder import ContextBuilder
from backend.app.retrieval.retriever import Retriever, RetrievalConfig
from backend.app.prompts.prompt_builder import PromptBuilder
from backend.app.services.llm_service import BaseLLMProvider
from backend.app.core.exceptions import EmptyQuestionException, NoChunksRetrievedException
from backend.app.core.telemetry import TelemetryCollector

logger = logging.getLogger(__name__)

class RAGService:
    """
    Coordinates semantic search, context building, prompt creation,
    and LLM inference into an observable RAG pipeline.
    """

    def __init__(
        self,
        retriever: Retriever,
        context_builder: ContextBuilder,
        prompt_builder: PromptBuilder,
        llm_service: BaseLLMProvider,
        query_rewriter = None
    ):
        self.retriever = retriever
        self.context_builder = context_builder
        self.prompt_builder = prompt_builder
        self.llm_service = llm_service
        self.query_rewriter = query_rewriter

    def answer(
        self,
        question: str,
        top_k: int = 5,
        history: list[dict] = None,
        retrieval_config: RetrievalConfig = None,
    ) -> dict:
        """
        Processes a user question, retrieves relevant code context,
        invokes the LLM generator, and records execution latencies.
        """
        request_id = str(uuid.uuid4())
        telemetry = TelemetryCollector(request_id)
        telemetry.start_stage("total")

        # 1. Validation
        cleaned_question = question.strip()
        if not cleaned_question:
            raise EmptyQuestionException("Question cannot be empty or whitespaces.")

        # Resolve retrieval config
        if not retrieval_config:
            retrieval_config = RetrievalConfig(
                top_k=top_k,
                fetch_k=20,
                use_mmr=True,
                lambda_mult=0.5,
                use_reranker=True,
                reranker_top_k=10,
                use_compressor=True
            )

        # 2. Query rewriting (if rewriter available and history exists)
        retrieval_query = cleaned_question
        if self.query_rewriter and history:
            telemetry.start_stage("query_rewrite")
            retrieval_query = self.query_rewriter.rewrite(cleaned_question, history)
            telemetry.end_stage()

        # 3. Semantic retrieval (vector search, MMR, and reranking are measured internally)
        results = self.retriever.retrieve(
            retrieval_query, config=retrieval_config, telemetry=telemetry
        )

        if not results:
            raise NoChunksRetrievedException(
                f"No relevant code chunks were retrieved for the query: '{retrieval_query}'"
            )

        # 4. Context Compression
        use_compress = retrieval_config.use_compressor
        results_for_context = []

        for r in results:
            rcopy = copy.copy(r)
            rcopy.payload = copy.copy(r.payload)
            
            orig_content = r.payload.get("content", "")
            orig_lines = len(orig_content.splitlines())
            orig_tokens = len(orig_content) // 4
            
            current_orig_lines = telemetry.counts["original_context_lines"]
            current_orig_tokens = telemetry.counts["original_context_tokens"]
            telemetry.set_count("original_context_lines", current_orig_lines + orig_lines)
            telemetry.set_count("original_context_tokens", current_orig_tokens + orig_tokens)

            if use_compress and self.retriever.compressor:
                telemetry.start_stage("compression")
                comp_content = self.retriever.compressor.compress(
                    retrieval_query, orig_content, rcopy.payload.get("path", "")
                )
                telemetry.end_stage()
                rcopy.payload["content"] = comp_content
            else:
                comp_content = orig_content

            comp_lines = len(comp_content.splitlines())
            comp_tokens = len(comp_content) // 4
            
            current_comp_lines = telemetry.counts["compressed_context_lines"]
            current_comp_tokens = telemetry.counts["compressed_context_tokens"]
            telemetry.set_count("compressed_context_lines", current_comp_lines + comp_lines)
            telemetry.set_count("compressed_context_tokens", current_comp_tokens + comp_tokens)

            results_for_context.append(rcopy)

        # 5. Context formatting and citation compilation
        telemetry.start_stage("prompt_build")
        context_string, citations = self.context_builder.build_context_and_citations(results_for_context)

        # 6. Prompt construction
        if history:
            messages = self.prompt_builder.build_rag_prompt_with_history(
                cleaned_question, context_string, history
            )
        else:
            messages = self.prompt_builder.build_rag_prompt(cleaned_question, context_string)

        prompt_str = "".join(m["content"] for m in messages)
        telemetry.set_count("prompt_tokens", len(prompt_str) // 4)
        telemetry.end_stage()

        # 7. LLM inference execution
        telemetry.start_stage("llm_generation")
        answer = self.llm_service.generate(messages)
        telemetry.end_stage()

        telemetry.set_count("completion_tokens", len(answer) // 4)
        
        # Stop total timer and export telemetry
        telemetry.end_stage() # total
        tel_data = telemetry.export()

        # Print structured telemetry logs
        logger.info(f"[TELEMETRY-RAG] {json.dumps(tel_data)}")

        response_payload = {
            "answer": answer,
            "citations": citations,
            "retrieved_chunks": len(results),
            "model": getattr(self.llm_service, "model_name", "unknown"),
            "latency_ms": tel_data["latencies"]["total_ms"],
            "metrics": {
                "retrieval_ms": tel_data["latencies"].get("vector_search_ms", 0),
                "context_ms": tel_data["latencies"].get("compression_ms", 0),
                "prompt_ms": tel_data["latencies"].get("prompt_build_ms", 0),
                "llm_ms": tel_data["latencies"].get("llm_generation_ms", 0),
                "total_ms": tel_data["latencies"]["total_ms"],
                # Extended telemetry breakdown
                "query_rewrite_ms": tel_data["latencies"].get("query_rewrite_ms", 0),
                "mmr_ms": tel_data["latencies"].get("mmr_ms", 0),
                "rerank_ms": tel_data["latencies"].get("rerank_ms", 0),
                "compression_ms": tel_data["latencies"].get("compression_ms", 0),
                "original_context_lines": tel_data["counts"]["original_context_lines"],
                "compressed_context_lines": tel_data["counts"]["compressed_context_lines"],
                "original_context_tokens": tel_data["counts"]["original_context_tokens"],
                "compressed_context_tokens": tel_data["counts"]["compressed_context_tokens"],
                "compression_ratio": tel_data["compression_ratio"],
                "prompt_tokens": tel_data["counts"]["prompt_tokens"],
                "completion_tokens": tel_data["counts"]["completion_tokens"],
                "candidates_retrieved": tel_data["counts"]["candidates_retrieved"],
                "after_mmr": tel_data["counts"]["after_mmr"],
                "after_reranker": tel_data["counts"]["after_reranker"],
                "pipeline_config": tel_data["config"],
                "average_scores": tel_data["scores"],
            }
        }
        if retrieval_query != cleaned_question:
            response_payload["rewritten_query"] = retrieval_query

        return response_payload

    def answer_stream(
        self,
        question: str,
        top_k: int = 5,
        request_id: str = None,
        history: list[dict] = None,
        retrieval_config: RetrievalConfig = None,
    ):
        """
        Streaming version of the RAG pipeline. Yields structured SSE event dicts.
        """
        if not request_id:
            request_id = str(uuid.uuid4())

        telemetry = TelemetryCollector(request_id)
        telemetry.start_stage("total")

        try:
            # 1. Validation
            cleaned_question = question.strip()
            if not cleaned_question:
                raise EmptyQuestionException("Question cannot be empty or whitespaces.")

            # Resolve retrieval config
            if not retrieval_config:
                retrieval_config = RetrievalConfig(
                    top_k=top_k,
                    fetch_k=20,
                    use_mmr=True,
                    lambda_mult=0.5,
                    use_reranker=True,
                    reranker_top_k=10,
                    use_compressor=True
                )

            # 2. Query rewriting (if rewriter available and history exists)
            retrieval_query = cleaned_question
            if self.query_rewriter and history:
                telemetry.start_stage("query_rewrite")
                retrieval_query = self.query_rewriter.rewrite(cleaned_question, history)
                telemetry.end_stage()

            # 3. Semantic retrieval (measured internally)
            results = self.retriever.retrieve(
                retrieval_query, config=retrieval_config, telemetry=telemetry
            )

            if not results:
                raise NoChunksRetrievedException(
                    f"No relevant code chunks were retrieved for the query: '{retrieval_query}'"
                )

            # 4. Context Compression
            use_compress = retrieval_config.use_compressor
            results_for_context = []

            for r in results:
                rcopy = copy.copy(r)
                rcopy.payload = copy.copy(r.payload)
                
                orig_content = r.payload.get("content", "")
                orig_lines = len(orig_content.splitlines())
                orig_tokens = len(orig_content) // 4
                
                current_orig_lines = telemetry.counts["original_context_lines"]
                current_orig_tokens = telemetry.counts["original_context_tokens"]
                telemetry.set_count("original_context_lines", current_orig_lines + orig_lines)
                telemetry.set_count("original_context_tokens", current_orig_tokens + orig_tokens)

                if use_compress and self.retriever.compressor:
                    telemetry.start_stage("compression")
                    comp_content = self.retriever.compressor.compress(
                        retrieval_query, orig_content, rcopy.payload.get("path", "")
                    )
                    telemetry.end_stage()
                    rcopy.payload["content"] = comp_content
                else:
                    comp_content = orig_content

                comp_lines = len(comp_content.splitlines())
                comp_tokens = len(comp_content) // 4
                
                current_comp_lines = telemetry.counts["compressed_context_lines"]
                current_comp_tokens = telemetry.counts["compressed_context_tokens"]
                telemetry.set_count("compressed_context_lines", current_comp_lines + comp_lines)
                telemetry.set_count("compressed_context_tokens", current_comp_tokens + comp_tokens)

                results_for_context.append(rcopy)

            # 5. Context & citations
            telemetry.start_stage("prompt_build")
            context_string, citations = self.context_builder.build_context_and_citations(results_for_context)

            # 6. Prompt construction
            if history:
                messages = self.prompt_builder.build_rag_prompt_with_history(
                    cleaned_question, context_string, history
                )
            else:
                messages = self.prompt_builder.build_rag_prompt(cleaned_question, context_string)

            prompt_str = "".join(m["content"] for m in messages)
            telemetry.set_count("prompt_tokens", len(prompt_str) // 4)
            telemetry.end_stage()

            # 7. Emit metadata event immediately (before LLM starts)
            metadata_payload = {
                "request_id": request_id,
                "citations": citations,
                "retrieved_chunks": len(results),
                "model": getattr(self.llm_service, "model_name", "unknown"),
                "retrieval_ms": telemetry.stages.get("vector_search_ms", 0),
                "context_ms": telemetry.stages.get("compression_ms", 0),
                "prompt_ms": telemetry.stages.get("prompt_build_ms", 0),
            }
            if retrieval_query != cleaned_question:
                metadata_payload["rewritten_query"] = retrieval_query

            yield ("metadata", metadata_payload)

            # 8. Stream LLM tokens
            telemetry.start_stage("llm_generation")
            tokens_generated = 0
            last_keepalive = time.perf_counter()
            KEEPALIVE_INTERVAL = 15.0  # seconds

            for chunk in self.llm_service.stream(messages):
                now = time.perf_counter()
                if now - last_keepalive >= KEEPALIVE_INTERVAL:
                    yield ("keepalive", {})
                    last_keepalive = now

                tokens_generated += 1
                yield ("token", {"text": chunk})

            telemetry.end_stage() # end llm_generation
            
            telemetry.set_count("completion_tokens", tokens_generated)
            
            telemetry.end_stage() # end total
            tel_data = telemetry.export()

            # Print structured telemetry logs
            logger.info(f"[TELEMETRY-RAG] {json.dumps(tel_data)}")

            tokens_per_second = round(
                tokens_generated / (tel_data["latencies"]["llm_generation_ms"] / 1000) 
                if tel_data["latencies"].get("llm_generation_ms", 0) > 0 else 0, 2
            )

            # 9. Emit final metrics (fully expanded breakdown)
            yield ("metrics", {
                "retrieval_ms": tel_data["latencies"].get("vector_search_ms", 0),
                "context_ms": tel_data["latencies"].get("compression_ms", 0),
                "prompt_ms": tel_data["latencies"].get("prompt_build_ms", 0),
                "llm_ms": tel_data["latencies"].get("llm_generation_ms", 0),
                "total_ms": tel_data["latencies"]["total_ms"],
                "tokens_generated": tokens_generated,
                "tokens_per_second": tokens_per_second,
                # Extended telemetry breakdown
                "query_rewrite_ms": tel_data["latencies"].get("query_rewrite_ms", 0),
                "mmr_ms": tel_data["latencies"].get("mmr_ms", 0),
                "rerank_ms": tel_data["latencies"].get("rerank_ms", 0),
                "compression_ms": tel_data["latencies"].get("compression_ms", 0),
                "original_context_lines": tel_data["counts"]["original_context_lines"],
                "compressed_context_lines": tel_data["counts"]["compressed_context_lines"],
                "original_context_tokens": tel_data["counts"]["original_context_tokens"],
                "compressed_context_tokens": tel_data["counts"]["compressed_context_tokens"],
                "compression_ratio": tel_data["compression_ratio"],
                "prompt_tokens": tel_data["counts"]["prompt_tokens"],
                "completion_tokens": tel_data["counts"]["completion_tokens"],
                "candidates_retrieved": tel_data["counts"]["candidates_retrieved"],
                "after_mmr": tel_data["counts"]["after_mmr"],
                "after_reranker": tel_data["counts"]["after_reranker"],
                "pipeline_config": tel_data["config"],
                "average_scores": tel_data["scores"],
            })

            # 10. Emit done signal
            yield ("done", {})

        except (EmptyQuestionException, NoChunksRetrievedException):
            raise
        except Exception as e:
            yield ("error", {
                "request_id": request_id,
                "message": str(e),
                "type": type(e).__name__
            })
