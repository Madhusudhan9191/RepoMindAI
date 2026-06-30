from fastapi import FastAPI

app = FastAPI(
    title="RepoMindAI",
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "message": "Welcome to RepoMindAI"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }