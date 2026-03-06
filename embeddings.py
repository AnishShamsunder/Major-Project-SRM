"""
embeddings.py
-------------
Loads the BAAI/bge-small-en-v1.5 sentence-transformer model and exposes a
helper function to generate embeddings for a list of texts.

The BGE models benefit from a short instruction prefix when encoding queries
(but NOT documents).  We handle that distinction here.
"""

import torch
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384  # output dimension of bge-small-en-v1.5

# BGE models use this prefix for query texts to improve retrieval quality.
# Document / passage texts are encoded WITHOUT the prefix.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# Use GPU if available, otherwise fall back to CPU.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Module-level singleton so the model is only loaded once per process.
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Return the cached SentenceTransformer model, loading it on first call."""
    global _model
    if _model is None:
        print(f"[embeddings] Loading model '{MODEL_NAME}' on {DEVICE.upper()} …")
        _model = SentenceTransformer(MODEL_NAME, device=DEVICE)
        print(f"[embeddings] Model loaded on {DEVICE.upper()}.")
    return _model


def embed_documents(texts: list[str], batch_size: int = 256) -> list[list[float]]:
    """
    Generate embeddings for a list of document/passage texts.

    Args:
        texts:      List of string passages to embed.
        batch_size: Number of texts per encoding batch (default 64).
                    Higher values use more RAM but reduce Python overhead.

    Returns:
        List of float vectors, one per input text.
    """
    model = get_model()
    # normalize_embeddings=True is recommended for cosine similarity
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return vectors.tolist()


def embed_query(query: str) -> list[float]:
    """
    Generate an embedding for a single user query string.

    The BGE instruction prefix is prepended to improve retrieval accuracy.

    Args:
        query: The user's natural-language question.

    Returns:
        A single float vector of length EMBEDDING_DIM.
    """
    model = get_model()
    # Prepend the BGE query instruction prefix
    prefixed_query = QUERY_INSTRUCTION + query
    vector = model.encode(prefixed_query, normalize_embeddings=True)
    return vector.tolist()
