import os

# Configure mock environment before importing the app to configure the Container
os.environ["LLM_PROVIDER"] = "mock"
os.environ["LLM_MODEL"] = "mock-gpt-4o"

from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_health():
    """Verify healthcheck endpoint returns healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_stats():
    """Verify statistics endpoint returns rich metadata."""
    response = client.get("/api/v1/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["collection"] == "repo_chunks"
    assert isinstance(data["vector_count"], int)
    assert "embedding_model" in data
    assert "llm_provider" in data
    assert data["llm_model"] == "mock-gpt-4o"
    assert "indexed_languages" in data
    assert data["dimension"] == 384


def test_ask():
    """Verify ask endpoint returns answer, metrics, and citations."""
    # Index the repository first to guarantee content exists in local db
    client.post("/api/v1/index", json={"repo_path": ".", "clear_existing": True})

    response = client.post(
        "/api/v1/ask",
        json={"question": "How are embeddings generated?", "top_k": 3}
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "[MOCK RESPONSE" in data["answer"]
    assert isinstance(data["citations"], list)
    assert len(data["citations"]) > 0
    assert data["retrieved_chunks"] == len(data["citations"])
    assert data["model"] == "mock-gpt-4o"
    assert isinstance(data["latency_ms"], int)
    
    # Check metric breakdown
    metrics = data["metrics"]
    assert "retrieval_ms" in metrics
    assert "context_ms" in metrics
    assert "prompt_ms" in metrics
    assert "llm_ms" in metrics
    assert "total_ms" in metrics


def test_ask_empty_question():
    """Verify EmptyQuestionException returns 400 Bad Request."""
    response = client.post("/api/v1/ask", json={"question": "   ", "top_k": 3})
    assert response.status_code == 400
    data = response.json()
    assert data["error_type"] == "EmptyQuestion"
    assert "Question cannot be empty" in data["detail"]


def test_clear_and_reindex():
    """Verify index clearing and re-indexing routes execute correctly."""
    # 1. Clear index
    response = client.delete("/api/v1/index")
    assert response.status_code == 200
    assert "cleared successfully" in response.json()["message"]

    # 2. Assert stats report 0 vectors
    response = client.get("/api/v1/stats")
    assert response.json()["vector_count"] == 0

    # 3. Ask question returns NoChunksRetrieved (404)
    response = client.post("/api/v1/ask", json={"question": "How to login?", "top_k": 3})
    assert response.status_code == 404
    assert response.json()["error_type"] == "NoChunksRetrieved"

    # 4. Ingest repository again
    response = client.post(
        "/api/v1/index",
        json={"repo_path": ".", "clear_existing": False}
    )
    assert response.status_code == 200
    metrics = response.json()
    assert metrics["python_files"] > 0
    assert metrics["chunks"] > 0
    assert metrics["vectors"] > 0
    assert isinstance(metrics["duration_ms"], int)

    # 5. Confirm stats report indexed vectors > 0
    response = client.get("/api/v1/stats")
    assert response.json()["vector_count"] > 0


def test_index_invalid_repo():
    """Verify RepositoryNotFoundException returns 404."""
    response = client.post(
        "/api/v1/index",
        json={"repo_path": "./invalid_folder_path_xyz", "clear_existing": False}
    )
    assert response.status_code == 404
    data = response.json()
    assert data["error_type"] == "RepositoryNotFound"
    assert "is not a valid directory" in data["detail"]


def test_settings():
    """Verify GET/POST LLM configuration settings routes."""
    # 1. Fetch current settings
    response = client.get("/api/v1/settings")
    assert response.status_code == 200
    data = response.json()
    assert data["provider"] == "mock"
    assert data["model"] == "mock-gpt-4o"

    # 2. Update config settings
    response = client.post(
        "/api/v1/settings",
        json={"provider": "mock", "model": "new-mock-gpt"}
    )
    assert response.status_code == 200
    assert response.json()["model"] == "new-mock-gpt"

    # 3. Confirm GET updates
    response = client.get("/api/v1/settings")
    assert response.json()["model"] == "new-mock-gpt"


def test_files():
    """Verify GET file inventory endpoint."""
    # 1. Fetch files for valid path
    response = client.get("/api/v1/files?repo_path=.")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert "path" in data[0]

    # 2. Fetch files for invalid path
    response = client.get("/api/v1/files?repo_path=./invalid_folder_path_xyz")
    assert response.status_code == 404
    assert response.json()["error_type"] == "RepositoryNotFound"


if __name__ == "__main__":
    print("Running integration tests manually...")
    try:
        test_health()
        print("test_health: PASSED")
        test_stats()
        print("test_stats: PASSED")
        test_ask()
        print("test_ask: PASSED")
        test_ask_empty_question()
        print("test_ask_empty_question: PASSED")
        test_clear_and_reindex()
        print("test_clear_and_reindex: PASSED")
        test_index_invalid_repo()
        print("test_index_invalid_repo: PASSED")
        test_settings()
        print("test_settings: PASSED")
        test_files()
        print("test_files: PASSED")
        print("\nAll integration tests PASSED successfully!")
    except AssertionError as e:
        print(f"\nAssertionError encountered: {str(e)}")
        import sys
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        import sys
        sys.exit(1)
