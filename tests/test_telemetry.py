import sys
from fastapi.testclient import TestClient
from backend.app.main import app, FEEDBACK_DB
from backend.app.core.container import Container
from backend.app.core.telemetry import TelemetryCollector

def run_tests():
    print("Running Telemetry and Observability tests...")

    # --- Test 1: TelemetryCollector utility functions ---
    collector = TelemetryCollector(request_id="test-123")
    assert collector.request_id == "test-123"
    
    # Assert counts initialization
    assert collector.counts["original_context_lines"] == 0
    
    # Add count values
    collector.set_count("original_context_lines", 50)
    collector.set_count("compressed_context_lines", 10)
    assert collector.counts["original_context_lines"] == 50
    assert collector.counts["compressed_context_lines"] == 10
    
    # Export check and compression ratio calculation
    data = collector.export()
    assert data["request_id"] == "test-123"
    # ratio = (50 - 10) / 50 = 40/50 = 0.8
    assert abs(data["compression_ratio"] - 0.8) < 1e-6
    print("Test 1 Passed: TelemetryCollector metric aggregation verified.")

    # --- Test 2: RAG Service Telemetry integration ---
    # Instantiate container and run an answer call to retrieve telemetry
    c = Container()
    ans = c.rag_service.answer("Where is parser initialized?")
    
    # Assert metrics contain all required telemetry fields
    assert "metrics" in ans
    metrics = ans["metrics"]
    
    # Latencies
    assert "total_ms" in metrics
    assert "retrieval_ms" in metrics
    assert "query_rewrite_ms" in metrics
    assert "mmr_ms" in metrics
    assert "rerank_ms" in metrics
    assert "compression_ms" in metrics
    assert "prompt_ms" in metrics
    assert "llm_ms" in metrics
    
    # Counts
    assert "original_context_lines" in metrics
    assert "compressed_context_lines" in metrics
    assert "original_context_tokens" in metrics
    assert "compressed_context_tokens" in metrics
    assert "compression_ratio" in metrics
    assert "prompt_tokens" in metrics
    assert "completion_tokens" in metrics
    assert "candidates_retrieved" in metrics
    assert "after_mmr" in metrics
    assert "after_reranker" in metrics
    
    # Configuration and Scores
    assert "pipeline_config" in metrics
    assert "average_scores" in metrics
    print("Test 2 Passed: RAG pipeline telemetry schema verification verified.")

    # --- Test 3: FastAPI Feedback Endpoint & Deduplication ---
    from backend.app.core.database import init_db
    init_db()
    client = TestClient(app)
    FEEDBACK_DB.clear()  # reset state

    # Register and login to obtain a valid JWT access token
    client.post("/api/v1/auth/register", json={
        "username": "telemetry_user",
        "password": "testpassword"
    })
    login_resp = client.post("/api/v1/auth/token", json={
        "username": "telemetry_user",
        "password": "testpassword"
    })
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # A: Success thumbs_up submission
    req_id = "request-uuid-abc"
    resp = client.post("/api/v1/feedback", json={
        "request_id": req_id,
        "rating": "thumbs_up",
        "feedback_text": "Very precise retrieval!"
    }, headers=headers)
    assert resp.status_code == 200
    res_data = resp.json()
    assert res_data["status"] == "success"
    assert res_data["is_overwrite"] is False
    
    # Verify in db
    assert req_id in FEEDBACK_DB
    assert FEEDBACK_DB[req_id]["rating"] == "thumbs_up"
    assert FEEDBACK_DB[req_id]["feedback_text"] == "Very precise retrieval!"
    
    # B: Duplicate check (overwrite rating and comments)
    resp_dup = client.post("/api/v1/feedback", json={
        "request_id": req_id,
        "rating": "thumbs_down",
        "feedback_text": "Changed my mind, too slow."
    }, headers=headers)
    assert resp_dup.status_code == 200
    res_data_dup = resp_dup.json()
    assert res_data_dup["status"] == "success"
    assert res_data_dup["is_overwrite"] is True  # correctly marked as overwrite
    assert FEEDBACK_DB[req_id]["rating"] == "thumbs_down"
    
    # C: Missing request ID validation error
    resp_err = client.post("/api/v1/feedback", json={
        "request_id": "",
        "rating": "thumbs_up"
    }, headers=headers)
    assert resp_err.status_code == 400
    print("Test 3 Passed: Feedback endpoint, validation, and overwrite logic verified.")

    # Clean up container connection
    try:
        from backend.app.core.container import get_container
        get_container().close()
    except Exception:
        pass

    print("\nAll Telemetry and Observability tests passed successfully!")

if __name__ == "__main__":
    try:
        run_tests()
    except AssertionError as e:
        import traceback
        traceback.print_exc()
        print(f"\nTEST FAILED: {str(e)}")
        try:
            from backend.app.core.container import get_container
            get_container().close()
        except Exception:
            pass
        sys.exit(1)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nUnexpected error during tests: {str(e)}")
        try:
            from backend.app.core.container import get_container
            get_container().close()
        except Exception:
            pass
        sys.exit(1)
