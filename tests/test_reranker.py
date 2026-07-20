import sys
from backend.app.retrieval.retriever import Retriever, RetrievalConfig
from backend.app.retrieval.reranker import RerankerService

class MockScoredPoint:
    def __init__(self, id, score, vector, payload=None):
        self.id = id
        self.score = score
        self.vector = vector
        self.payload = payload or {}

class MockEmbeddingService:
    def embed_text(self, text):
        return [1.0, 0.0]
    def get_embedding_dimension(self):
        return 2

class MockQdrantService:
    def __init__(self, candidates):
        self.candidates = candidates
    def search(self, query, limit=5, with_vectors=False):
        # Return candidates up to limit
        return self.candidates[:limit]

def run_tests():
    print("Running Cross-Encoder Reranking tests...")

    # --- Test 1: Lexical overlap calculation in RerankerService ---
    reranker = RerankerService(model_name="mock-model")
    
    # 50% overlap of unique query words ("scanner" and "parser" -> only "scanner" in document)
    overlap1 = reranker._calculate_lexical_overlap("scanner parser", "the scanner initializes here")
    assert abs(overlap1 - 0.5) < 1e-6, f"Expected 0.5, got {overlap1}"

    # 100% overlap
    overlap2 = reranker._calculate_lexical_overlap("scanner", "scanner scanner scanner")
    assert abs(overlap2 - 1.0) < 1e-6, f"Expected 1.0, got {overlap2}"

    # 0% overlap
    overlap3 = reranker._calculate_lexical_overlap("parser", "scanner scanner")
    assert abs(overlap3 - 0.0) < 1e-6, f"Expected 0.0, got {overlap3}"
    print("Test 1 Passed: Lexical overlap calculation verified.")

    # --- Test 2: Mock Reranking rescores and reshuffles candidates ---
    # Query is "initialize scanner"
    # Point 1: content "import scanner" (score: 0.90, overlap: 0.5)
    # Point 2: content "scanner.initialize() method" (score: 0.85, overlap: 1.0)
    # Reranker score calculation: score + 0.1 * overlap + 0.0001 * len(content)
    # Point 1 mock score: 0.90 + 0.05 + 0.0014 = 0.9514
    # Point 2 mock score: 0.85 + 0.10 + 0.0028 = 0.9528
    # Point 2 should rank FIRST after reranking despite lower original vector score.
    p1 = MockScoredPoint(1, 0.90, [1.0, 0.0], {"content": "import scanner"})
    p2 = MockScoredPoint(2, 0.85, [1.0, 0.0], {"content": "scanner.initialize() method"})
    
    candidates = [p1, p2]
    reranked = reranker.rerank("initialize scanner", candidates)
    
    assert len(reranked) == 2
    assert reranked[0].id == 2, f"Expected Candidate 2 to rank first, got {reranked[0].id}"
    assert reranked[1].id == 1
    assert "rerank_score" in reranked[0].payload
    assert "rerank_score" in reranked[1].payload
    print("Test 2 Passed: Reshuffling and mock reranker scores verified.")

    # --- Test 3: Small candidate pool ---
    # If candidate pool is small (e.g. 2 candidates) but reranker_top_k=10, it should execute without index errors
    mock_es = MockEmbeddingService()
    mock_qd = MockQdrantService(candidates=[p1, p2])
    retriever = Retriever(embedding_service=mock_es, qdrant_service=mock_qd, reranker_service=reranker)
    
    config_small = RetrievalConfig(top_k=2, fetch_k=20, use_mmr=True, use_reranker=True, reranker_top_k=10)
    results = retriever.retrieve("initialize scanner", config=config_small)
    assert len(results) == 2
    print("Test 3 Passed: Small candidate pool execution verified.")

    # --- Test 4: Retriever pipeline combinations ---
    # Case A: MMR=False, Reranker=True
    config_rerank_only = RetrievalConfig(top_k=1, use_mmr=False, use_reranker=True, reranker_top_k=2)
    results_a = retriever.retrieve("initialize scanner", config=config_rerank_only)
    assert len(results_a) == 1
    assert results_a[0].id == 2  # still reranked to first place
    
    # Case B: MMR=True, Reranker=False
    # Point 1 and 2 vectors are identical, so second point will get penalized by MMR
    # With MMR-only, Candidate 1 should stay first because Candidate 2 gets heavily penalized
    config_mmr_only = RetrievalConfig(top_k=2, use_mmr=True, lambda_mult=0.5, use_reranker=False)
    results_b = retriever.retrieve("initialize scanner", config=config_mmr_only)
    assert len(results_b) == 2
    assert results_b[0].id == 1  # candidate 1 ranks first
    print("Test 4 Passed: Retriever pipeline branching (Rerank-only vs MMR-only) verified.")

    print("\nAll Cross-Encoder Reranking tests passed successfully!")

if __name__ == "__main__":
    try:
        run_tests()
    except AssertionError as e:
        print(f"\nTEST FAILED: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error during tests: {str(e)}")
        sys.exit(1)
