# RepoMind AI — Codebase Retrieval & RAG Platform

RepoMind AI is a production-grade, state-of-the-art codebase retrieval and reasoning assistant (RAG). It enables developers to securely index, search, and chat with multi-language codebases and technical documentation with ultra-low latency and verifiable accuracy.

Built with a modular, multi-stage retrieval architecture, RepoMind AI delivers high-precision answers with file-level and line-level citations, native JWT authentication, and a containerized deployment stack.

---

## 🚀 Key Features

* **Hybrid Multi-Language Code Chunking Engine**:
  * **AST Syntax Parsers**: Extracts structural function, class, and module blocks for Python (`FunctionDef`, `AsyncFunctionDef`, `ClassDef`).
  * **Language-Aware Sliding Window**: Line-sliding chunking for JavaScript (`.js`, `.jsx`), TypeScript (`.ts`, `.tsx`), Markdown (`.md`), JSON, YAML, Dockerfiles, Go, Rust, Java, and shell scripts.
  * **Fault-Tolerant Fallback**: Syntax error resilient fallback chunker for broken or malformed code files.

* **Incremental Ingestion & SHA-256 Deletion Sync**:
  * **Hash Manifest Tracking**: Computes SHA-256 file hashes (`.file_hashes.json`) to skip re-embedding unchanged files during incremental indexing runs.
  * **Deletion & Mutation Sync**: Automatically purges old Qdrant vector points for modified or deleted files using point payload path filters (`delete_chunks_by_path`).
  * **Batch Processing**: Multi-file chunk batch inference and batch Qdrant upserts.

* **Hybrid Search & Explicit Reranking Engine**:
  * **Dense Vector Search**: Qdrant vector index for semantic similarity.
  * **Lexical BM25 Search**: In-memory Okapi BM25 keyword retriever for exact code symbol and variable matching.
  * **Candidate Union & Reranking**: Merges dense and lexical candidate pools and computes explicit hybrid scores:
    $$\text{hybrid\_score} = \alpha \cdot \text{dense\_similarity} + \beta \cdot \text{normalized\_bm25} + \gamma \cdot \text{symbol\_overlap}$$
  * **Structure-Aware Context Compression**: Weight-scores individual code lines, expands logical margins, and dynamically filters filler lines to fit tight prompt token budgets.

* **Quantitative RAG Evaluation & Benchmark Harness**:
  * Built-in evaluation framework (`evaluation/eval_suite.py`) testing Precision@5, Recall@5, MRR, Context Compression Ratio %, and stage latencies.

* **FastAPI Lifespan & Telemetry**:
  * `@asynccontextmanager` lifespan handler for clean startup DB initialization and graceful resource shutdown.
  * Decoupled telemetry tracking for query LRU embedding cache hits/misses, candidate counts, and stage latency breakdowns.

* **Production Security & JWT Auth**:
  * PBKDF2 bcrypt password hashing.
  * Short-lived bearer access tokens (15-min expiry) and rotated refresh tokens (30-day expiry).
  * SHA-256 database hashed refresh tokens to protect session integrity against database compromise.

* **NAT-Safe Sliding-Window Rate Limiter**:
  * Custom in-memory sliding window rate limits.
  * Keys requests by authenticated `user.id` when logged in, falling back to client IP address for anonymous users.

* **SSE Stream Orchestration**:
  * Progressive token rendering and keepalive loops to prevent browser buffer chunk truncation.

---

## 🛠️ Technology Stack

* **Backend**: FastAPI (Python 3.11), Uvicorn.
* **Vector DB**: Qdrant (Local / Server).
* **Search & Lexical**: In-memory Okapi BM25, Sentence-Transformers.
* **Database**: SQLite (configured in Write-Ahead Log (WAL) mode for concurrent read/write performance).
* **Cryptography**: `PyJWT`, native `bcrypt`.
* **Frontend**: React, Vite, Vanilla CSS.
* **Containers**: Docker, Docker Compose, Nginx.

---

## 📁 Directory Structure

```text
RepoMindAi/
├── .github/workflows/   # CI/CD pipelines
├── backend/
│   └── app/
│       ├── chunking/     # Hybrid multi-language AST + sliding line window chunkers
│       ├── core/         # DB, rate limit, security, container, telemetry
│       ├── embedding/    # Thread-safe query LRU cache & embedding service
│       ├── ingestion/    # Scanner, file filters, SHA-256 incremental pipelines
│       ├── parsers/      # Python AST parsers
│       ├── prompts/      # RAG prompt template builder
│       ├── retrieval/    # BM25Retriever, MMR, ContextCompressor, Hybrid Reranker, Retriever orchestration
│       ├── services/     # LLM services, RAG orchestrator, Query rewriter
│       ├── vectorstore/  # Qdrant client & batch point deletion services
│       └── main.py       # FastAPI router & lifespan context manager
├── docker/
│   ├── backend.Dockerfile
│   ├── frontend.Dockerfile
│   └── nginx.conf        # SSE proxy buffering off config
├── evaluation/           # Quantitative RAG evaluation & benchmark suite
├── frontend/             # Vite React client
├── tests/                # Automated pytest & standalone test suites
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

## 📊 Running Quantitative RAG Benchmarks

To run the automated quantitative evaluation harness measuring Precision@5, Recall@5, MRR, Context Compression Ratio %, and stage latencies across all retrieval configurations:

```bash
python -m evaluation.eval_suite
```

---

## 🧪 Testing & Verification

Run the full pytest test suite across all 9 test modules:

```powershell
python -m pytest tests/ --tb=short
```

You can also run individual feature verification scripts:

```powershell
python -m tests.test_chunker
python -m tests.test_production
python -m tests.test_telemetry
python -m tests.test_compressor
python -m tests.test_reranker
python -m tests.test_mmr
python -m tests.test_query_rewriter
python -m tests.test_eval
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
