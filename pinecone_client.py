"""
pinecone_client.py
------------------
Manages the connection to Pinecone and ensures the target index exists.

Required environment variables (loaded from a .env file via python-dotenv):
    PINECONE_API_KEY    – your Pinecone API key
    PINECONE_INDEX_NAME – name of the index to create / use
"""

import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from embeddings import EMBEDDING_DIM

# ---------------------------------------------------------------------------
# Load environment variables from .env (no-op if already set in environment)
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Constants – read from environment
# ---------------------------------------------------------------------------
PINECONE_API_KEY: str = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX_NAME: str = os.environ["PINECONE_INDEX_NAME"]

# Index settings
METRIC = "cosine"
# Serverless spec – uses the free "starter" tier; adjust cloud/region as needed.
CLOUD = "aws"
REGION = "us-east-1"

# Module-level Pinecone client / index singletons
_pc: Pinecone | None = None
_index = None  # pinecone.Index


def get_pinecone_client() -> Pinecone:
    """Return the cached Pinecone client, creating it on first call."""
    global _pc
    if _pc is None:
        print("[pinecone] Initialising Pinecone client …")
        _pc = Pinecone(api_key=PINECONE_API_KEY)
        print("[pinecone] Client ready.")
    return _pc


def get_or_create_index():
    """
    Return a handle to the Pinecone index, creating it first if it does not
    already exist.

    The index is configured with:
        - dimension : EMBEDDING_DIM (384 for bge-small-en-v1.5)
        - metric    : cosine
    """
    global _index
    if _index is not None:
        return _index

    pc = get_pinecone_client()
    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if PINECONE_INDEX_NAME not in existing_indexes:
        print(f"[pinecone] Index '{PINECONE_INDEX_NAME}' not found – creating …")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric=METRIC,
            spec=ServerlessSpec(cloud=CLOUD, region=REGION),
        )
        print(f"[pinecone] Index '{PINECONE_INDEX_NAME}' created.")
    else:
        print(f"[pinecone] Index '{PINECONE_INDEX_NAME}' already exists – reusing.")

    _index = pc.Index(PINECONE_INDEX_NAME)
    return _index


def upsert_vectors(vectors: list[dict]) -> dict:
    """
    Upsert a batch of vectors into the Pinecone index.

    Each item in `vectors` must be a dict with keys:
        id       (str)         – unique identifier for the chunk
        values   (list[float]) – the embedding vector
        metadata (dict)        – arbitrary key-value metadata

    Args:
        vectors: List of vector dicts (see above).

    Returns:
        The upsert response from Pinecone (contains 'upserted_count').
    """
    index = get_or_create_index()

    # Pinecone recommends batch sizes of ~100 for optimal throughput
    BATCH_SIZE = 100
    total_upserted = 0

    for i in range(0, len(vectors), BATCH_SIZE):
        batch = vectors[i : i + BATCH_SIZE]
        response = index.upsert(vectors=batch)
        # Pinecone 8.x returns UpsertResponse (object), not a dict
        total_upserted += getattr(response, "upserted_count", len(batch))
        print(f"[pinecone] Upserted batch {i // BATCH_SIZE + 1} "
              f"({len(batch)} vectors, running total: {total_upserted})")

    return {"upserted_count": total_upserted}


def query_index(vector: list[float], top_k: int = 5) -> dict:
    """
    Query the Pinecone index for the most similar vectors.

    Args:
        vector: The query embedding (list of floats).
        top_k:  Number of nearest neighbours to return.

    Returns:
        The raw Pinecone query response dict containing 'matches'.
    """
    index = get_or_create_index()
    response = index.query(
        vector=vector,
        top_k=top_k,
        include_metadata=True,  # return chunk text + source in results
    )
    return response
