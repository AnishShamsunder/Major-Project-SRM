"""
retrieval.py
------------
Query pipeline: embed the user's question and retrieve the most relevant
chunks from Pinecone.
"""

from embeddings import embed_query
from pinecone_client import query_index

# Default number of results to retrieve
DEFAULT_TOP_K = 5


def retrieve(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """
    Retrieve the most relevant document chunks for a user query.

    Pipeline:
        1. Embed the query string using the BGE model (with instruction prefix).
        2. Query the Pinecone index for the nearest `top_k` vectors.
        3. Parse and return the matches as a clean list of result dicts.

    Args:
        query:  Natural-language question from the user.
        top_k:  How many results to return (default 5).

    Returns:
        List of dicts, each containing:
            {
                "id"       : str,   – Pinecone vector ID
                "score"    : float, – cosine similarity score (0–1)
                "text"     : str,   – the raw chunk text
                "source"   : str,   – original filename
                "chunk_idx": int,   – position of chunk in the source document
            }
    """
    # ── Step 1: Embed the query ──────────────────────────────────────────────
    print(f"[retrieval] Embedding query: '{query[:80]}…'" if len(query) > 80 else f"[retrieval] Embedding query: '{query}'")
    query_vector = embed_query(query)

    # ── Step 2: Query Pinecone ───────────────────────────────────────────────
    print(f"[retrieval] Querying Pinecone for top {top_k} matches …")
    response = query_index(vector=query_vector, top_k=top_k)

    # ── Step 3: Parse results ────────────────────────────────────────────────
    # Pinecone 8.x returns typed objects (QueryResponse / ScoredVector),
    # not plain dicts – use attribute access instead of .get() / [key].
    results: list[dict] = []
    matches = getattr(response, "matches", None) or []
    for match in matches:
        metadata = getattr(match, "metadata", None) or {}
        results.append({
            "id": match.id,
            "score": round(float(match.score), 6),
            "text": metadata.get("text", ""),
            "source": metadata.get("source", "unknown"),
            "chunk_idx": int(metadata.get("chunk_idx", -1)),
        })

    print(f"[retrieval] Retrieved {len(results)} result(s).")
    return results
