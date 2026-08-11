import time
import os
from collections import defaultdict
from typing import Optional
from fastapi import Request, HTTPException, status, Depends
from backend.app.core.security import get_current_user

class RateLimiter:
    """
    Sliding window log rate limiter implemented in-memory.
    """

    def __init__(self, requests_limit: int, window_seconds: int = 60):
        self.requests_limit = requests_limit
        self.window_seconds = window_seconds
        self.history = defaultdict(list)

    def check(self, key: str):
        """
        Validates the request limit log for the key. Raises HTTP 429 if exceeded.
        """
        now = time.time()
        cutoff = now - self.window_seconds
        
        # Clean up stale timestamps outside the window
        self.history[key] = [ts for ts in self.history[key] if ts > cutoff]
        
        if len(self.history[key]) >= self.requests_limit:
            oldest_ts = self.history[key][0]
            retry_after = max(1, int((oldest_ts + self.window_seconds) - now))
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later.",
                headers={"Retry-After": str(retry_after)}
            )
        
        self.history[key].append(now)

def resolve_rate_limit_key(request: Optional[Request], current_user: Optional[dict] = None) -> str:
    """
    Keys the request by authenticated user ID if available, falling back to client IP.
    """
    if current_user and "id" in current_user:
        return f"user_{current_user['id']}"
    
    # Fallback to client IP
    client_ip = "unknown_ip"
    if request and hasattr(request, "client") and request.client and request.client.host:
        client_ip = request.client.host
    return f"ip_{client_ip}"


# Read limits from environment variables (with fallback defaults)
RATE_LIMIT_RAG = int(os.getenv("RATE_LIMIT_RAG", "100"))
RATE_LIMIT_AUTH = int(os.getenv("RATE_LIMIT_AUTH", "20"))

rag_limiter = RateLimiter(requests_limit=RATE_LIMIT_RAG, window_seconds=60)
auth_limiter = RateLimiter(requests_limit=RATE_LIMIT_AUTH, window_seconds=60)

def limit_rag_requests(request: Request):
    """
    FastAPI dependency to rate limit RAG operations.
    """
    key = resolve_rate_limit_key(request, None)
    rag_limiter.check(key)

def limit_auth_requests(request: Request):
    """
    FastAPI dependency to rate limit auth operations.
    """
    key = resolve_rate_limit_key(request, None)
    auth_limiter.check(key)
