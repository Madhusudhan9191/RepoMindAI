import sys
from backend.app.retrieval.compressor import ContextCompressor
from backend.app.retrieval.retriever import Retriever, RetrievalConfig

class MockScoredPoint:
    def __init__(self, id, score, vector, payload=None):
        self.id = id
        self.score = score
        self.vector = vector
        self.payload = payload or {}

class MockEmbeddingService:
    def embed_text(self, text):
        return [1.0]

class MockQdrantService:
    def __init__(self, points):
        self.points = points
    def search(self, query, limit=5, with_vectors=False):
        return self.points[:limit]

def run_tests():
    print("Running Context Compression tests...")

    compressor = ContextCompressor()

    # --- Test 1: Query token extraction and Camel/snake splits ---
    words = compressor._extract_query_words("Where is RepositoryScanner initialized?")
    assert "repository" in words
    assert "scanner" in words
    assert "initialized" in words
    assert "where" not in words  # keyword (ignored if added to blacklist or filtered)
    
    words_splits = compressor._extract_query_words("initialize_scanner")
    assert "initialize" in words_splits
    assert "scanner" in words_splits
    print("Test 1 Passed: Query token splitting (CamelCase/snake_case) verified.")

    # --- Test 2: Scoring lines (exact vs split, structural def/class boosts) ---
    query_words = {"scanner", "initialize"}
    # Standard query token match
    score1 = compressor._score_line("scanner = None", query_words)
    assert score1 >= 2
    # Split token match
    score2 = compressor._score_line("initialize_repository()", query_words)
    assert score2 >= 1
    # Structural definition matching query token should get extra weight
    score3 = compressor._score_line("def initialize_scanner():", query_words)
    assert score3 > score1, f"Expected definition to have higher score than normal match: {score3} vs {score1}"
    print("Test 2 Passed: Line scoring weights verified.")

    # --- Test 3: Parent scope mapping and window expansion ---
    code_block = (
        "class Pipeline:\n"               # line 0 (parent class)
        "    def run(self):\n"            # line 1 (parent def)
        "        print('hello')\n"         # line 2
        "        print('world')\n"         # line 3
        "        print('debugging')\n"     # line 4
        "        scanner.initialize()\n"   # line 5 (matches query word 'initialize')
        "        print('done')\n"          # line 6
        "        print('finished')\n"      # line 7
        "        print('completed')\n"     # line 8
        "        print('exited')\n"        # line 9
        "        print('1')\n"             # line 10
        "        print('2')\n"             # line 11
        "        print('3')\n"             # line 12
        "        print('4')\n"             # line 13
        "        print('5')\n"             # line 14
        "        print('6')\n"             # line 15
    )
    # The chunk is 16 lines (>= 15 lines safety guard).
    # Line 5 contains 'initialize'. Focus window index 5 +/- 2 -> lines 3, 4, 5, 6, 7 are kept.
    # Pre-calculated scopes: parent class is 0 (Pipeline), parent def is 1 (run).
    # These should be added to kept indices.
    # Lines 8-15 should be omitted since they are outside window and not scopes.
    compressed = compressor.compress("initialize", code_block, "pipeline.py")
    
    assert "class Pipeline:" in compressed
    assert "def run(self):" in compressed
    assert "scanner.initialize()" in compressed
    assert "print('completed')" not in compressed
    assert "# ... lines 9-16 omitted for brevity" in compressed or "# ... lines 9-16 omitted" in compressed
    print("Test 3 Passed: Window expansion and scope preservation verified.")

    # --- Test 4: Language-aware comment markers ---
    marker_py = compressor._get_omission_marker("test.py")
    assert marker_py == "#"
    marker_js = compressor._get_omission_marker("test.js")
    assert marker_js == "//"
    marker_html = compressor._get_omission_marker("test.html")
    assert marker_html == "<!-- {} -->"
    print("Test 4 Passed: File-extension comment markers verified.")

    # --- Test 5: Safety guards (short files and minimal compression) ---
    # Chunks < 15 lines are skipped
    short_code = "print('hello')\n" * 5
    compressed_short = compressor.compress("hello", short_code, "test.py")
    assert len(compressed_short.splitlines()) == 5
    
    # Chunk where no matching word is found -> returned unchanged
    long_code = "print('unrelated')\n" * 20
    compressed_nomatch = compressor.compress("hello", long_code, "test.py")
    assert len(compressed_nomatch.splitlines()) == 20
    print("Test 5 Passed: Safety guards and fallback scenarios verified.")

    # --- Test 6: Integration via Retriever.get_context() ---
    payload = {
        "id": "1",
        "path": "test.py",
        "name": "Pipeline",
        "type": "class",
        "start_line": 1,
        "end_line": 16,
        "content": code_block,
    }
    mock_point = MockScoredPoint(1, 0.9, [1.0], payload)
    mock_es = MockEmbeddingService()
    mock_qd = MockQdrantService([mock_point])
    
    retriever = Retriever(
        embedding_service=mock_es,
        qdrant_service=mock_qd,
        compressor=compressor
    )
    
    config = RetrievalConfig(use_compressor=True)
    ctx = retriever.get_context("initialize", config=config)
    
    assert "class Pipeline:" in ctx
    assert "# ... lines" in ctx
    print("Test 6 Passed: Retriever get_context integration verified.")

    print("\nAll Context Compression tests passed successfully!")

if __name__ == "__main__":
    try:
        run_tests()
    except AssertionError as e:
        print(f"\nTEST FAILED: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error during tests: {str(e)}")
        sys.exit(1)
