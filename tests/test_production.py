import sys
import time
import jwt
from fastapi import status
from fastapi.testclient import TestClient

from backend.app.main import app, FEEDBACK_DB
from backend.app.core.database import init_db, get_db_connection
from backend.app.core.security import JWT_SECRET, hash_refresh_token
from backend.app.core.rate_limit import rag_limiter, auth_limiter

def run_tests():
    print("Running Production Platform and Security tests...")
    init_db()  # Ensure database tables exist before clearing
    client = TestClient(app)

    # Clean up SQLite and start with fresh test users
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM refresh_tokens;")
        cursor.execute("DELETE FROM users;")
        conn.commit()
    finally:
        conn.close()

    # --- Test 1: Health & Readiness Probes ---
    resp_health = client.get("/health")
    assert resp_health.status_code == 200
    assert resp_health.json() == {"status": "healthy"}
    print("Test 1a Passed: /health check verified.")

    resp_ready = client.get("/ready")
    assert resp_ready.status_code == 200
    assert resp_ready.json() == {"status": "ready"}
    print("Test 1b Passed: /ready check verified.")

    # --- Test 2: Ingestion & Invalidation of Unsecured Requests ---
    # Attempting to query metrics without bearer tokens must fail
    resp_stats_anon = client.get("/api/v1/stats")
    assert resp_stats_anon.status_code == status.HTTP_401_UNAUTHORIZED
    print("Test 2 Passed: Anonymous access blocked on protected routes.")

    # --- Test 3: Registration & Authentication Flows ---
    # Register test user
    resp_reg = client.post("/api/v1/auth/register", json={
        "username": "coder_bob",
        "password": "securepassword123"
    })
    assert resp_reg.status_code == 200
    assert resp_reg.json()["status"] == "success"

    # Attempt duplicate registration
    resp_reg_dup = client.post("/api/v1/auth/register", json={
        "username": "coder_bob",
        "password": "anotherpassword"
    })
    assert resp_reg_dup.status_code == 400
    assert resp_reg_dup.json()["detail"] == "Username already registered"
    print("Test 3a Passed: User registration and duplicate checks verified.")

    # Login - Success
    resp_login = client.post("/api/v1/auth/token", json={
        "username": "coder_bob",
        "password": "securepassword123"
    })
    assert resp_login.status_code == 200
    login_data = resp_login.json()
    assert "access_token" in login_data
    assert "refresh_token" in login_data
    assert login_data["token_type"] == "bearer"
    
    access_token = login_data["access_token"]
    refresh_token = login_data["refresh_token"]

    # Login - Bad Password
    resp_login_fail = client.post("/api/v1/auth/token", json={
        "username": "coder_bob",
        "password": "wrongpassword"
    })
    assert resp_login_fail.status_code == 401
    print("Test 3b Passed: Authentication and credential checks verified.")

    # --- Test 4: Protected API requests with JWT ---
    # Make request with valid JWT
    headers = {"Authorization": f"Bearer {access_token}"}
    resp_stats = client.get("/api/v1/stats", headers=headers)
    assert resp_stats.status_code == 200
    assert resp_stats.json()["status"] == "healthy"
    
    # Make request with malformed token
    bad_headers = {"Authorization": "Bearer malformedtokenhere"}
    resp_bad = client.get("/api/v1/stats", headers=bad_headers)
    assert resp_bad.status_code == 401
    assert "validate credentials" in resp_bad.json()["detail"]

    # Make request with expired token
    expired_payload = {
        "sub": "coder_bob",
        "user_id": "some-uuid",
        "exp": int(time.time() - 3600)  # expired 1 hour ago
    }
    expired_token = jwt.encode(expired_payload, JWT_SECRET, algorithm="HS256")
    expired_headers = {"Authorization": f"Bearer {expired_token}"}
    
    resp_expired = client.get("/api/v1/stats", headers=expired_headers)
    assert resp_expired.status_code == 401
    assert "expired" in resp_expired.json()["detail"].lower()
    print("Test 4 Passed: Token verification, malformed token blocks, and expiry rules verified.")

    # --- Test 5: Refresh token rotation and revocation ---
    # Rotate token using valid refresh token
    resp_ref = client.post("/api/v1/auth/refresh", json={"refresh_token": refresh_token})
    assert resp_ref.status_code == 200
    ref_data = resp_ref.json()
    assert "access_token" in ref_data
    assert "refresh_token" in ref_data
    
    new_access_token = ref_data["access_token"]
    new_refresh_token = ref_data["refresh_token"]

    # Access protected route using the rotated fresh access token
    new_headers = {"Authorization": f"Bearer {new_access_token}"}
    resp_new_stats = client.get("/api/v1/stats", headers=new_headers)
    assert resp_new_stats.status_code == 200

    # Request refresh again using the OLD refresh token (should fail since rotated and deleted)
    resp_ref_old = client.post("/api/v1/auth/refresh", json={"refresh_token": refresh_token})
    assert resp_ref_old.status_code == 401
    assert "Invalid or revoked" in resp_ref_old.json()["detail"]

    # Request refresh with invalid string
    resp_ref_bad = client.post("/api/v1/auth/refresh", json={"refresh_token": "some-fake-refresh-string"})
    assert resp_ref_bad.status_code == 401
    print("Test 5 Passed: Refresh token rotation and revocation checks verified.")

    # --- Test 6: Rate Limiting sliding window ---
    # Reset limiter statistics
    rag_limiter.history.clear()
    
    # Trigger RAG rate limiter: Default threshold is 10 requests / min.
    # Make 10 requests. All should pass.
    for i in range(10):
        resp_rag = client.post("/api/v1/ask", json={
            "question": "what is this?",
            "top_k": 1,
            "history": []
        }, headers=new_headers)
        assert resp_rag.status_code in (200, 404)
        
    # The 11th request should be blocked with 429
    resp_rag_blocked = client.post("/api/v1/ask", json={
        "question": "what is this?",
        "top_k": 1,
        "history": []
    }, headers=new_headers)
    assert resp_rag_blocked.status_code == 429
    assert "Rate limit exceeded" in resp_rag_blocked.json()["detail"]
    assert "Retry-After" in resp_rag_blocked.headers
    print("Test 6 Passed: Sliding window log rate limiting and HTTP 429 throttles verified.")

    # Clean up container connection before completing
    try:
        from backend.app.core.container import get_container
        get_container().close()
    except Exception:
        pass

    print("\nAll Production Platform and Security tests passed successfully!")

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
