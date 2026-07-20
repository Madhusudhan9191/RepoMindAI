import sys
from backend.app.retrieval.mmr import cosine_similarity, maximum_marginal_relevance

class MockScoredPoint:
    """Mocks Qdrant ScoredPoint structures for testing."""
    def __init__(self, id, score, vector, payload=None):
        self.id = id
        self.score = score
        self.vector = vector
        self.payload = payload or {}

def run_tests():
    print("Running MMR Diversification tests...")

    # --- Test 1: Cosine Similarity math ---
    # Colinear vectors (angle 0) -> similarity 1.0
    v1 = [1.0, 2.0, 3.0]
    v2 = [2.0, 4.0, 6.0]
    sim1 = cosine_similarity(v1, v2)
    assert abs(sim1 - 1.0) < 1e-6, f"Expected 1.0, got {sim1}"

    # Orthogonal vectors (angle 90) -> similarity 0.0
    v3 = [1.0, 0.0, 0.0]
    v4 = [0.0, 1.0, 0.0]
    sim2 = cosine_similarity(v3, v4)
    assert abs(sim2 - 0.0) < 1e-6, f"Expected 0.0, got {sim2}"

    # Opposite vectors (angle 180) -> similarity -1.0
    v5 = [1.0, 0.0]
    v6 = [-1.0, 0.0]
    sim3 = cosine_similarity(v5, v6)
    assert abs(sim3 - (-1.0)) < 1e-6, f"Expected -1.0, got {sim3}"
    
    print("Test 1 Passed: Cosine Similarity math calculation verified.")

    # --- Test 2: Standard MMR selection ---
    # Query vector is [1.0, 0.0, 0.0] (along X axis)
    # Candidate 1: relevance 0.90, vector [0.99, 0.01, 0.0] (highly similar to query)
    # Candidate 2: relevance 0.85, vector [0.98, 0.02, 0.0] (highly similar to query and Candidate 1)
    # Candidate 3: relevance 0.70, vector [0.0, 1.0, 0.0]    (orthogonal to query, very diverse)
    query = [1.0, 0.0, 0.0]
    candidates = [
        MockScoredPoint(1, 0.90, [0.99, 0.01, 0.0]),
        MockScoredPoint(2, 0.85, [0.98, 0.02, 0.0]),
        MockScoredPoint(3, 0.70, [0.0, 1.0, 0.0]),
    ]
    # With top_k=2 and lambda=0.5:
    # First selected: Candidate 1 (score 0.90)
    # Remaining: Candidate 2 and Candidate 3.
    # Scores for next selection:
    # Candidate 2: 0.5 * 0.85 - 0.5 * cos(Cand2, Cand1) = 0.425 - 0.5 * ~0.999 = ~ -0.075
    # Candidate 3: 0.5 * 0.70 - 0.5 * cos(Cand3, Cand1) = 0.35 - 0.5 * ~0.01 = ~ 0.345
    # Candidate 3 should be chosen second even though Candidate 2 has higher raw relevance score.
    selected = maximum_marginal_relevance(query, candidates, top_k=2, lambda_mult=0.5)
    assert len(selected) == 2
    assert selected[0].id == 1
    assert selected[1].id == 3, f"Expected Candidate 3 to be selected for diversity, got: {selected[1].id}"
    assert "mmr_score" in selected[0].payload
    assert "mmr_score" in selected[1].payload
    print("Test 2 Passed: Standard MMR selection and diversification verified.")

    # --- Test 3: Identical vectors stability ---
    # If all candidate vectors are identical, MMR should be stable and return top_k candidates in order of relevance
    candidates_ident = [
        MockScoredPoint(1, 0.95, [1.0, 0.0]),
        MockScoredPoint(2, 0.90, [1.0, 0.0]),
        MockScoredPoint(3, 0.85, [1.0, 0.0]),
    ]
    selected_ident = maximum_marginal_relevance([1.0, 0.0], candidates_ident, top_k=2, lambda_mult=0.5)
    assert len(selected_ident) == 2
    assert selected_ident[0].id == 1
    assert selected_ident[1].id == 2
    print("Test 3 Passed: Identical vectors selection stability verified.")

    # --- Test 4: Orthogonal vectors stability ---
    # If candidates are already orthogonal, MMR should follow standard ranking since diversity is already high
    candidates_ortho = [
        MockScoredPoint(1, 0.90, [1.0, 0.0, 0.0]),
        MockScoredPoint(2, 0.80, [0.0, 1.0, 0.0]),
        MockScoredPoint(3, 0.70, [0.0, 0.0, 1.0]),
    ]
    selected_ortho = maximum_marginal_relevance([1.0, 0.0, 0.0], candidates_ortho, top_k=2, lambda_mult=0.5)
    assert len(selected_ortho) == 2
    assert selected_ortho[0].id == 1
    assert selected_ortho[1].id == 2
    print("Test 4 Passed: Orthogonal vectors selection stability verified.")

    # --- Test 5: Fallback and boundary conditions ---
    # Empty candidates
    assert maximum_marginal_relevance([1.0], [], top_k=5) == []
    # top_k larger than candidate pool size
    cand_pool = [MockScoredPoint(1, 0.9, [1.0])]
    assert len(maximum_marginal_relevance([1.0], cand_pool, top_k=10)) == 1
    print("Test 5 Passed: Fallback and boundaries verified.")

    print("\nAll MMR tests passed successfully!")

if __name__ == "__main__":
    try:
        run_tests()
    except AssertionError as e:
        print(f"\nTEST FAILED: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error during tests: {str(e)}")
        sys.exit(1)
