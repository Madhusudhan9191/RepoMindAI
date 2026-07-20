# RepoMind AI — Codebase Retrieval & RAG Platform

RepoMind AI is a production-grade, state-of-the-art codebase retrieval and reasoning assistant (RAG). It enables developers to securely index, search, and chat with their repositories and technical documentation. 

Built with a modular, multi-stage retrieval architecture, RepoMind AI delivers high-precision answers with file-level and line-level citations, native JWT authentication, and a containerized deployment stack.

---

## 🚀 Key Features

* **Multi-Stage Codebase Retrieval Stack**:
  * **Semantic Recall**: Batch ingestion and dense vector search via Qdrant.
  * **MMR (Maximal Marginal Relevance)**: Diversifies code chunk results to eliminate redundant file context.
  * **Lexical Overlap Reranking**: Re-ranks candidates by true textual matching scores to maximize relevance.
  * **Structure-Aware Context Compression**: Weight-scores individual code lines, expands logical margins, and dynamically filters out filler tokens to fit code context into tight prompt limits.
* **Production Security & JWT Auth**:
  * PBKDF2 bcrypt password hashing.
  * Short-lived bearer access tokens (15-min expiry) and rotated refresh tokens (30-day expiry).
  * SHA-256 database hashed refresh tokens to protect session integrity against database compromise.
* **NAT-Safe Sliding-Window Rate Limiter**:
  * Custom in-memory sliding window rate limits.
  * Keys requests by authenticated `user.id` when logged in, falling back to client IP address for anonymous users (preventing NAT address blocking).
* **SSE Stream Orchestration**:
  * Progressive token rendering and keepalive loops to prevent browser buffer chunk truncation.
  * Graceful thread teardowns via FastAPI shutdown hooks.
* **Observability & Feedback Loop**:
  * Decoupled telemetry logging capturing latency and token ratios per request.
  * Deduplicated feedback submissions allowing ratings and reviews overwrites.

---

## 🛠️ Technology Stack

* **Backend**: FastAPI (Python 3.11), Uvicorn.
* **Vector DB**: Qdrant (Local / Server).
* **Database**: SQLite (configured in Write-Ahead Log (WAL) mode for concurrent read/write performance).
* **Cryptography**: `PyJWT`, native `bcrypt` (independent of unmaintained third-party wrappers).
* **Frontend**: React, Vite, Vanilla CSS.
* **Containers**: Docker, Docker Compose, Nginx.

---

## 📁 Directory Structure

```text
RepoMindAi/
├── .github/workflows/   # CI/CD pipelines
├── backend/
│   └── app/
│       ├── chunking/     # AST-based code chunking
│       ├── core/         # DB, rate limit, security, container, telemetry
│       ├── embedding/    # Sentence-transformers embedding service
│       ├── ingestion/    # Scanner, gitignore filters, pipelines
│       ├── parsers/      # Python AST parsers
│       ├── prompts/      # RAG prompt template builder
│       ├── retrieval/    # MMR, Compressor, Reranker, Retriever orchestration
│       ├── services/     # LLM services, RAG orchestrator, Query rewriter
│       ├── vectorstore/  # Qdrant client configurations
│       └── main.py       # FastAPI router configurations
├── docker/
│   ├── backend.Dockerfile
│   ├── frontend.Dockerfile
│   └── nginx.conf        # SSE proxy buffering off config
├── frontend/             # Vite React client
├── tests/                # 6 Sequential test suites
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## ⚙️ Quick Start

### 1. Host Installation
Create a Python virtual environment and install dependencies:
```bash
# In project root
python -m venv backend/.venv
.\backend\.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. Environment Variables Configuration
Copy the `.env.example` file and configure parameters:
```bash
cp .env.example .env
```
Ensure your environment contains correct keys:
- `JWT_SECRET`: Secure encryption key.
- `QDRANT_URL`: Vector database URI (e.g. `http://localhost:6333`).
- `LLM_PROVIDER`: `mock`, `openai`, `ollama`, or `gemini`.

---

## 🧪 Testing & Verification

RepoMind AI runs 6 sequential integration and regression test suites. Execute them in your host shell:

```powershell
$env:PYTHONPATH="."
.\backend\.venv\Scripts\python tests/test_production.py
.\backend\.venv\Scripts\python tests/test_telemetry.py
.\backend\.venv\Scripts\python tests/test_compressor.py
.\backend\.venv\Scripts\python tests/test_reranker.py
.\backend\.venv\Scripts\python tests/test_mmr.py
.\backend\.venv\Scripts\python tests/test_query_rewriter.py
```

---

## 🐳 Docker Compose Deployment

Run the entire containerized stack (Qdrant, Backend, and Frontend reverse proxied via Nginx) locally:

```bash
docker-compose up --build
```
- **React Frontend**: `http://localhost/`
- **FastAPI API Docs**: `http://localhost/api/docs`
- **Qdrant DB Dashboard**: `http://localhost:6333/dashboard`
