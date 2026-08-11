import json
import logging
import datetime
import uuid
from typing import Literal, Optional
from fastapi import FastAPI, HTTPException, Request, Depends, status
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from backend.app.core.container import get_container
from backend.app.core.database import init_db, get_db_connection
from backend.app.core.security import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    get_current_user,
    hash_refresh_token
)
from backend.app.core.rate_limit import limit_rag_requests, limit_auth_requests
from backend.app.core.exceptions import (
    RepoMindAIException,
    RepositoryNotFoundException,
    EmptyQuestionException,
    NoChunksRetrievedException,
    LLMTimeoutException,
)
from backend.app.ingestion.scanner import RepositoryScanner
from backend.app.ingestion.ingest_pipeline import IngestionPipeline

from fastapi.middleware.cors import CORSMiddleware

from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager handling startup database initialization and shutdown resource cleanup."""
    init_db()
    yield
    try:
        container = get_container()
        container.close()
    except Exception as e:
        logger.error(f"Error closing container on shutdown: {str(e)}")

app = FastAPI(
    title="RepoMindAI",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# FastAPI Exception Handlers
# --------------------------------------------------

@app.exception_handler(RepositoryNotFoundException)
async def repository_not_found_handler(request: Request, exc: RepositoryNotFoundException):
    return JSONResponse(
        status_code=404,
        content={"detail": str(exc), "error_type": "RepositoryNotFound"}
    )

@app.exception_handler(EmptyQuestionException)
async def empty_question_handler(request: Request, exc: EmptyQuestionException):
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc), "error_type": "EmptyQuestion"}
    )

@app.exception_handler(NoChunksRetrievedException)
async def no_chunks_retrieved_handler(request: Request, exc: NoChunksRetrievedException):
    return JSONResponse(
        status_code=404,
        content={"detail": str(exc), "error_type": "NoChunksRetrieved"}
    )

@app.exception_handler(LLMTimeoutException)
async def llm_timeout_handler(request: Request, exc: LLMTimeoutException):
    return JSONResponse(
        status_code=503,
        content={"detail": str(exc), "error_type": "LLMTimeout"}
    )

@app.exception_handler(RepoMindAIException)
async def repomind_generic_handler(request: Request, exc: RepoMindAIException):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "error_type": "InternalRepoMindError"}
    )

# --------------------------------------------------
# Request & Response Schemas
# --------------------------------------------------

class ConversationTurn(BaseModel):
    """A single turn in the conversation history."""
    role: Literal["user", "assistant"] = Field(..., description="Speaker role.")
    content: str = Field(..., description="Message text content.")


class QueryRequest(BaseModel):
    question: str = Field(..., description="The question related to the repository codebase.")
    top_k: int = Field(5, description="Number of relevant chunks to retrieve.")
    history: list[ConversationTurn] = Field(
        default_factory=list,
        description="Prior conversation turns for multi-turn context. Empty = stateless query."
    )

class Citation(BaseModel):
    file: str
    function: str
    type: str
    start_line: int
    end_line: int
    score: float

class MetricBreakdown(BaseModel):
    retrieval_ms: int
    context_ms: int
    prompt_ms: int
    llm_ms: int
    total_ms: int
    # Extended telemetry fields
    query_rewrite_ms: int | None = 0
    mmr_ms: int | None = 0
    rerank_ms: int | None = 0
    compression_ms: int | None = 0
    original_context_lines: int | None = 0
    compressed_context_lines: int | None = 0
    original_context_tokens: int | None = 0
    compressed_context_tokens: int | None = 0
    compression_ratio: float | None = 0.0
    prompt_tokens: int | None = 0
    completion_tokens: int | None = 0
    candidates_retrieved: int | None = 0
    after_mmr: int | None = 0
    after_reranker: int | None = 0
    pipeline_config: dict | None = None
    average_scores: dict | None = None

class FeedbackRequest(BaseModel):
    request_id: str = Field(..., description="Unique request ID from response metadata.")
    rating: Literal["thumbs_up", "thumbs_down"] = Field(..., description="User feedback rating.")
    feedback_text: str | None = Field(None, description="Optional text feedback.")

class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    retrieved_chunks: int
    model: str
    latency_ms: int
    metrics: MetricBreakdown

class IndexRequest(BaseModel):
    repo_path: str = Field(".", description="Local directory path to index.")
    clear_existing: bool = Field(True, description="Clear existing database collection before ingestion.")

class IndexResponse(BaseModel):
    files_scanned: int
    python_files: int
    chunks: int
    vectors: int
    duration_ms: int

class StatsResponse(BaseModel):
    status: str
    collection: str
    vector_count: int
    embedding_model: str
    llm_provider: str
    llm_model: str
    indexed_languages: list[str]
    dimension: int

class SettingsRequest(BaseModel):
    provider: str = Field(..., description="LLM provider name: mock, ollama, openai, gemini.")
    model: str = Field(..., description="Model identifier name.")
    api_key: str | None = Field(None, description="Optional API key for OpenAI or Gemini.")
    api_base: str | None = Field(None, description="Optional API base URL for Ollama.")

class SettingsResponse(BaseModel):
    provider: str
    model: str
    api_base: str | None = None

# Auth Schemas
class RegisterRequest(BaseModel):
    username: str = Field(..., description="Desired unique username.")
    password: str = Field(..., description="Plaintext password.")

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"

class RefreshRequest(BaseModel):
    refresh_token: str

# --------------------------------------------------
# Endpoints
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "Welcome to RepoMindAI API",
        "docs_url": "/docs",
        "api_v1_prefix": "/api/v1"
    }


@app.get("/health")
def health():
    """Viability probe checks if application container runs."""
    return {
        "status": "healthy"
    }


@app.get("/ready")
def ready():
    """Readiness probe checking Qdrant, embeddings, SQLite database."""
    container = get_container()
    try:
        # Check SQLite db connection
        conn = get_db_connection()
        conn.execute("SELECT 1;")
        conn.close()

        # Check embedding and Qdrant readiness
        container.qdrant_service.get_collection_info()
        
        return {"status": "ready"}
    except Exception as e:
        logger.error(f"Readiness probe failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Services not ready: {str(e)}"
        )


# --- Authentication Routes ---

@app.post("/api/v1/auth/register", dependencies=[Depends(limit_auth_requests)])
def register(payload: RegisterRequest):
    """Registers a new user inside the SQLite database."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        
        # Check if username exists
        cursor.execute("SELECT 1 FROM users WHERE username = ?", (payload.username,))
        if cursor.fetchone():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already registered"
            )
            
        hashed = hash_password(payload.password)
        user_id = str(uuid.uuid4())
        created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        cursor.execute(
            "INSERT INTO users (id, username, hashed_password, created_at) VALUES (?, ?, ?, ?)",
            (user_id, payload.username, hashed, created_at)
        )
        conn.commit()
        return {"status": "success", "message": "User registered successfully."}
    finally:
        conn.close()


@app.post("/api/v1/auth/token", response_model=TokenResponse, dependencies=[Depends(limit_auth_requests)])
def login(payload: RegisterRequest):
    """Generates short-lived Access Token and long-lived Refresh Token."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id, hashed_password FROM users WHERE username = ?", (payload.username,))
        row = cursor.fetchone()
        if not row or not verify_password(payload.password, row["hashed_password"]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )
            
        user_id = row["id"]
        access_token = create_access_token(user_id, payload.username)
        refresh_token = create_refresh_token(user_id, db_conn=conn)
        
        conn.commit()
        return TokenResponse(access_token=access_token, refresh_token=refresh_token)
    finally:
        conn.close()


@app.post("/api/v1/auth/refresh", response_model=TokenResponse, dependencies=[Depends(limit_auth_requests)])
def refresh(payload: RefreshRequest):
    """Rotates refresh tokens, validating SHA-256 hashes and expiry."""
    token_hash = hash_refresh_token(payload.refresh_token)
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT r.user_id, r.expires_at, u.username FROM refresh_tokens r "
            "JOIN users u ON r.user_id = u.id WHERE r.token_hash = ?",
            (token_hash,)
        )
        row = cursor.fetchone()
        if not row:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or revoked refresh token"
            )
            
        # Verify expiry
        expires_at = datetime.datetime.fromisoformat(row["expires_at"])
        if expires_at < datetime.datetime.now(datetime.timezone.utc):
            cursor.execute("DELETE FROM refresh_tokens WHERE token_hash = ?", (token_hash,))
            conn.commit()
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Refresh token expired"
            )
            
        user_id = row["user_id"]
        username = row["username"]
        
        # De-authorize old refresh token (rotate refresh tokens)
        cursor.execute("DELETE FROM refresh_tokens WHERE token_hash = ?", (token_hash,))
        
        # Generate new tokens
        access_token = create_access_token(user_id, username)
        refresh_token = create_refresh_token(user_id, db_conn=conn)
        
        conn.commit()
        return TokenResponse(access_token=access_token, refresh_token=refresh_token)
    finally:
        conn.close()


# --- Secured Application Routes ---

@app.get("/api/v1/stats", response_model=StatsResponse, dependencies=[Depends(get_current_user)])
def get_stats():
    try:
        container = get_container()
        
        # Query vector store stats
        collection_info = container.qdrant_service.get_collection_info()
        vector_count = collection_info.points_count
        
        return StatsResponse(
            status="healthy",
            collection=container.qdrant_service.COLLECTION_NAME,
            vector_count=vector_count,
            embedding_model=container.embedding_service.get_model_name(),
            llm_provider=type(container.llm_service).__name__,
            llm_model=getattr(container.llm_service, "model_name", "unknown"),
            indexed_languages=list(set(RepositoryScanner.LANGUAGE_MAP.values())),
            dimension=container.embedding_service.get_embedding_dimension()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve stats: {str(e)}")


@app.post("/api/v1/ask", response_model=QueryResponse, dependencies=[Depends(limit_rag_requests)])
def ask(payload: QueryRequest):
    container = get_container()
    history = [t.model_dump() for t in payload.history] if payload.history else None
    response_data = container.rag_service.answer(
        question=payload.question,
        top_k=payload.top_k,
        history=history,
    )
    return response_data


# Simple in-memory tracker for feedback deduplication check
FEEDBACK_DB = {}

@app.post("/api/v1/feedback")
def submit_feedback(payload: FeedbackRequest):
    feedback_logger = logging.getLogger("backend.app.feedback")
    
    req_id = payload.request_id.strip()
    if not req_id:
        raise HTTPException(status_code=400, detail="Missing or invalid request_id")
        
    now_str = datetime.datetime.now(datetime.timezone.utc).isoformat()
    
    existed = req_id in FEEDBACK_DB
    FEEDBACK_DB[req_id] = {
        "rating": payload.rating,
        "feedback_text": payload.feedback_text,
        "timestamp": now_str
    }
    
    log_msg = {
        "event": "user_feedback",
        "request_id": req_id,
        "rating": payload.rating,
        "comment": payload.feedback_text,
        "timestamp": now_str,
        "is_overwrite": existed
    }
    
    feedback_logger.info(f"[TELEMETRY-FEEDBACK] {json.dumps(log_msg)}")
    
    return {
        "status": "success",
        "message": "Feedback submitted successfully.",
        "is_overwrite": existed
    }


@app.post("/api/v1/ask/stream", dependencies=[Depends(limit_rag_requests)])
def ask_stream(payload: QueryRequest):
    """
    Streaming RAG endpoint.
    """
    container = get_container()
    history = [t.model_dump() for t in payload.history] if payload.history else None

    def event_generator():
        try:
            for event_type, payload_data in container.rag_service.answer_stream(
                question=payload.question,
                top_k=payload.top_k,
                history=history,
            ):
                yield f"event: {event_type}\ndata: {json.dumps(payload_data)}\n\n"
        except (EmptyQuestionException, NoChunksRetrievedException) as exc:
            error_payload = {
                "message": str(exc),
                "type": type(exc).__name__
            }
            yield f"event: error\ndata: {json.dumps(error_payload)}\n\n"
        except Exception as exc:
            error_payload = {
                "message": f"Unexpected server error: {str(exc)}",
                "type": type(exc).__name__
            }
            yield f"event: error\ndata: {json.dumps(error_payload)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


@app.post("/api/v1/index", response_model=IndexResponse)
def run_indexing(payload: IndexRequest):
    container = get_container()
    
    if payload.clear_existing:
        print("Clearing Qdrant vector store collection...")
        container.qdrant_service.clear()
        
    pipeline = IngestionPipeline(payload.repo_path)
    metrics = pipeline.ingest()
    return metrics


@app.delete("/api/v1/index")
def clear_index():
    try:
        container = get_container()
        container.qdrant_service.clear()
        return {
            "message": "Vector store index cleared successfully."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear index: {str(e)}")


@app.get("/api/v1/settings", response_model=SettingsResponse)
def get_settings():
    try:
        container = get_container()
        provider_name = type(container.llm_service).__name__
        provider_map = {
            "MockLLMProvider": "mock",
            "OllamaProvider": "ollama",
            "OpenAIProvider": "openai",
            "GeminiProvider": "gemini"
        }
        provider = provider_map.get(provider_name, "unknown")
        model = getattr(container.llm_service, "model_name", "unknown")
        api_base = getattr(container.llm_service, "api_base", None)
        
        return SettingsResponse(
            provider=provider,
            model=model,
            api_base=api_base
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/settings", response_model=SettingsResponse)
def update_settings(payload: SettingsRequest):
    try:
        container = get_container()
        container.configure_llm(
            provider=payload.provider,
            model=payload.model,
            api_key=payload.api_key,
            api_base=payload.api_base
        )
        
        provider_name = type(container.llm_service).__name__
        provider_map = {
            "MockLLMProvider": "mock",
            "OllamaProvider": "ollama",
            "OpenAIProvider": "openai",
            "GeminiProvider": "gemini"
        }
        provider = provider_map.get(provider_name, "unknown")
        model = getattr(container.llm_service, "model_name", "unknown")
        api_base = getattr(container.llm_service, "api_base", None)
        
        return SettingsResponse(
            provider=provider,
            model=model,
            api_base=api_base
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update LLM configuration: {str(e)}")


@app.get("/api/v1/files")
def get_files(repo_path: str = "."):
    try:
        from backend.app.ingestion.scanner import RepositoryScanner
        scanner = RepositoryScanner(repo_path)
        inventory = scanner.scan()
        return inventory
    except Exception as e:
        if "not a valid directory" in str(e) or "system cannot find" in str(e).lower() or "does not exist" in str(e).lower():
            raise RepositoryNotFoundException(f"Repository path '{repo_path}' is not a valid directory.")
        raise HTTPException(status_code=500, detail=str(e))