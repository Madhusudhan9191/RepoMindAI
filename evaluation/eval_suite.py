import time
import json
from pathlib import Path
from backend.app.core.container import get_container
from backend.app.retrieval.retriever import RetrievalConfig
from backend.app.ingestion.ingest_pipeline import IngestionPipeline


# Ground-truth evaluation dataset mapping natural language queries to expected file targets
EVAL_DATASET = [
    {
        "query": "How are embeddings generated for code chunks?",
        "expected_files": ["backend/app/embedding/embedding_service.py"]
    },
    {
        "query": "Where is sliding window rate limiting implemented?",
        "expected_files": ["backend/app/core/rate_limit.py"]
    },
    {
        "query": "How are JWT access tokens created and verified?",
        "expected_files": ["backend/app/core/security.py"]
    },
    {
        "query": "Where is the FastAPI startup and shutdown lifecycle handled?",
        "expected_files": ["backend/app/main.py"]
    },
    {
        "query": "How does the Python AST parser extract function metadata?",
        "expected_files": ["backend/app/parsers/python_parser.py"]
    },
    {
        "query": "Where is maximum marginal relevance diversification calculated?",
        "expected_files": ["backend/app/retrieval/mmr.py"]
    },
    {
        "query": "How does the ContextCompressor compress lines and calculate scores?",
        "expected_files": ["backend/app/retrieval/compressor.py"]
    },
    {
        "query": "Where is Qdrant vector collection created and queried?",
        "expected_files": ["backend/app/vectorstore/qdrant_service.py"]
    }
]


def evaluate_pipeline(container, config: RetrievalConfig, stage_name: str) -> dict:
    """
    Evaluates a specific retrieval pipeline configuration against the ground-truth benchmark dataset.
    """
    total_queries = len(EVAL_DATASET)
    precision_sum = 0.0
    recall_sum = 0.0
    mrr_sum = 0.0
    total_retrieval_ms = 0
    total_orig_lines = 0
    total_comp_lines = 0

    for item in EVAL_DATASET:
        query = item["query"]
        expected = set(f.lower() for f in item["expected_files"])

        start_t = time.perf_counter()
        results = container.retriever.retrieve(query=query, top_k=config.top_k, config=config)
        retrieval_ms = int((time.perf_counter() - start_t) * 1000)
        total_retrieval_ms += retrieval_ms

        retrieved_files = [r.payload.get("path", "").lower() for r in results if r.payload]

        # Calculate Precision@K and Recall@K
        hits = [f for f in retrieved_files if f in expected]
        num_hits = len(set(hits))

        p_at_k = num_hits / len(retrieved_files) if retrieved_files else 0.0
        r_at_k = num_hits / len(expected) if expected else 0.0

        precision_sum += p_at_k
        recall_sum += r_at_k

        # Calculate Reciprocal Rank (RR)
        rr = 0.0
        for idx, f in enumerate(retrieved_files, start=1):
            if f in expected:
                rr = 1.0 / idx
                break
        mrr_sum += rr

        # Context Compression Ratio
        if config.use_compressor:
            raw_context = container.retriever.get_context(query=query, top_k=config.top_k, config=config)
            total_comp_lines += len(raw_context.splitlines())
            # Estimate uncompressed lines
            for r in results:
                total_orig_lines += len(r.payload.get("content", "").splitlines())

    avg_p = precision_sum / total_queries
    avg_r = recall_sum / total_queries
    avg_mrr = mrr_sum / total_queries
    avg_latency = total_retrieval_ms / total_queries
    compression_ratio = 100.0 * (1.0 - (total_comp_lines / total_orig_lines)) if (total_orig_lines > 0 and config.use_compressor) else 0.0

    return {
        "stage": stage_name,
        "precision_at_5": round(avg_p, 4),
        "recall_at_5": round(avg_r, 4),
        "mrr": round(avg_mrr, 4),
        "compression_pct": round(compression_ratio, 1),
        "avg_latency_ms": round(avg_latency, 1)
    }


def run_benchmark():
    print("=" * 80)
    print("REPOMIND AI QUANTITATIVE RAG BENCHMARK & EVALUATION HARNESS")
    print("=" * 80)

    # 1. Initialize container & index repository
    container = get_container()
    print("\nEnsuring repository index is up to date...")
    pipeline = IngestionPipeline(".")
    pipeline.ingest()

    configs = [
        (RetrievalConfig(top_k=5, use_mmr=False, use_reranker=False, use_compressor=False), "1. Baseline Dense (Qdrant Only)"),
        (RetrievalConfig(top_k=5, use_mmr=True, use_reranker=False, use_compressor=False), "2. Dense + MMR Diversification"),
        (RetrievalConfig(top_k=5, use_mmr=True, use_reranker=True, use_compressor=False), "3. + Hybrid Reranker (Dense+BM25+Symbol)"),
        (RetrievalConfig(top_k=5, use_mmr=True, use_reranker=True, use_compressor=True), "4. + Structure-Aware Compression"),
    ]

    results_table = []
    for cfg, stage_name in configs:
        print(f"\nEvaluating Stage: {stage_name}...")
        metrics = evaluate_pipeline(container, cfg, stage_name)
        results_table.append(metrics)

    print("\n" + "=" * 80)
    print(f"{'Pipeline Stage':<42} | {'Precision@5':<11} | {'Recall@5':<9} | {'MRR':<6} | {'Compress %':<10} | {'Avg Latency'}")
    print("-" * 80)

    for res in results_table:
        print(
            f"{res['stage']:<42} | "
            f"{res['precision_at_5']:<11.4f} | "
            f"{res['recall_at_5']:<9.4f} | "
            f"{res['mrr']:<6.4f} | "
            f"{res['compression_pct']:<9.1f}% | "
            f"{res['avg_latency_ms']} ms"
        )
    print("=" * 80 + "\n")
    return results_table


if __name__ == "__main__":
    run_benchmark()
