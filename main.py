"""
main.py
-------
FastAPI server that exposes two endpoints:

    POST /ingest
        Reads the data/ folder, generates embeddings for every chunk,
        and upserts them into Pinecone.

    POST /query
        Accepts { "query": "...", "top_k": 5 } and returns the most
        relevant context chunks retrieved from Pinecone.

Run:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Environment variables required (see .env.example):
    PINECONE_API_KEY
    PINECONE_INDEX_NAME
"""

from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Load .env before any module that reads environment variables
load_dotenv()

from ingest import ingest, DATA_DIR
from retrieval import retrieve
from gemini_client import generate_answer, GEMINI_MODEL

# ---------------------------------------------------------------------------
# Lifespan: warm-up tasks on startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Pre-load the embedding model and verify the Pinecone index exists at
    startup so the first request is not slow.
    """
    print("[main] Warming up embedding model …")
    from embeddings import get_model
    get_model()  # triggers the model download / cache load

    print("[main] Verifying Pinecone index …")
    from pinecone_client import get_or_create_index
    get_or_create_index()

    print("[main] Initialising Gemini client …")
    from gemini_client import get_client
    get_client()

    print("[main] Server ready.")
    yield  # hand control back to FastAPI
    print("[main] Shutting down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RAG Backend – Drug Interaction Knowledge Base",
    description=(
        "Retrieval-Augmented Generation backend powered by "
        "BAAI/bge-small-en-v1.5 embeddings and Pinecone vector search."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow all origins for local development (tighten for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class IngestResponse(BaseModel):
    message: str
    files_processed: int
    total_chunks: int
    upserted_count: int


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural-language question")
    top_k: int = Field(5, ge=1, le=20, description="Number of results to return")


class ChunkResult(BaseModel):
    id: str
    score: float
    text: str
    source: str
    chunk_idx: int


class QueryResponse(BaseModel):
    query: str
    results: list[ChunkResult]


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural-language question about drug interactions")
    top_k: int = Field(5, ge=1, le=20, description="Number of context chunks to retrieve")


class AskResponse(BaseModel):
    query: str
    answer: str
    model: str
    sources: list[ChunkResult]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/ingest", response_model=IngestResponse, summary="Ingest data folder into Pinecone")
async def ingest_endpoint():
    """
    **POST /ingest**

    Reads every file under the `data/` directory, splits them into overlapping
    chunks (~400–500 tokens), generates embeddings with BAAI/bge-small-en-v1.5,
    and upserts the vectors + metadata into the configured Pinecone index.

    This endpoint is idempotent: re-running it will overwrite existing vectors
    with the same deterministic IDs rather than creating duplicates.

    Returns a summary with the number of files processed, total chunks
    created, and vectors upserted.
    """
    if not DATA_DIR.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Data directory not found: {DATA_DIR}",
        )

    try:
        summary = ingest(DATA_DIR)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

    return IngestResponse(
        message="Ingestion complete.",
        files_processed=summary["files_processed"],
        total_chunks=summary["total_chunks"],
        upserted_count=summary["upserted_count"],
    )


@app.post("/query", response_model=QueryResponse, summary="Retrieve relevant context chunks")
async def query_endpoint(request: QueryRequest):
    """
    **POST /query**

    Accepts a JSON body with a `query` string (and optional `top_k`) and
    returns the most semantically similar document chunks from Pinecone.

    Example request body:
    ```json
    {
        "query": "What are the interactions between warfarin and aspirin?",
        "top_k": 5
    }
    ```

    Each result includes the chunk text, cosine similarity score, source
    filename, and chunk index.
    """
    try:
        results = retrieve(query=request.query, top_k=request.top_k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {exc}") from exc

    return QueryResponse(
        query=request.query,
        results=[ChunkResult(**r) for r in results],
    )


@app.post("/ask", response_model=AskResponse, summary="Retrieve context and generate a Gemini answer")
async def ask_endpoint(request: AskRequest):
    """
    **POST /ask**

    Full RAG pipeline: retrieves the top-k most relevant chunks from Pinecone,
    then feeds them as context to Gemini 2.5 Flash Lite to generate a
    grounded natural-language answer.

    Example request body:
    ```json
    {
        "query": "What happens if I take warfarin and aspirin together?",
        "top_k": 5
    }
    ```

    Returns the generated answer, the model used, and the source chunks.
    """
    # Step 1: retrieve relevant chunks
    try:
        results = retrieve(query=request.query, top_k=request.top_k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {exc}") from exc

    if not results:
        raise HTTPException(status_code=404, detail="No relevant context found in the knowledge base.")

    # Step 2: generate answer with Gemini
    context_chunks = [r["text"] for r in results]
    try:
        answer = generate_answer(query=request.query, context_chunks=context_chunks)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc

    return AskResponse(
        query=request.query,
        answer=answer,
        model=GEMINI_MODEL,
        sources=[ChunkResult(**r) for r in results],
    )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health", summary="Health check")
async def health():
    """Returns 200 OK when the server is running."""
    return {"status": "ok"}
