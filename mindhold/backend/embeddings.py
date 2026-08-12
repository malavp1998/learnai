"""
Embeddings via the Jina AI HTTP API — the "API-based embedding" counterpart
to ../../06_rag_basics.py, which instead runs a HuggingFace model locally.

Local model (06_rag_basics.py): downloads weights once, runs on YOUR machine,
no network call per request, no API key.

API-based model (this file): no download, no local compute — every call to
embed() makes an HTTP request to Jina's servers, they run the model, and send
back the vectors. Costs API credits, needs internet + JINA_API_KEY, but keeps
your machine light and lets you swap embedding models by changing a string.

This is the same trade-off as ChatGroq (API) vs. running a local LLM — just
one level down, at the embedding step of the RAG pipeline.
"""

from __future__ import annotations

import httpx

from settings import settings

JINA_API_URL = "https://api.jina.ai/v1/embeddings"
JINA_MODEL = "jina-embeddings-v3"
EMBEDDING_DIM = 1024  # jina-embeddings-v3's default output size — must match
                       # the VECTOR(1024) column dimension in db.py


async def embed(texts: list[str], task: str = "retrieval.passage") -> list[list[float]]:
    """
    Turn a list of text strings into a list of embedding vectors (one vector
    per input string, order preserved).

    `task` tells Jina what the embedding is FOR — it nudges the model to
    produce vectors tuned for that use. We use "retrieval.passage" when
    embedding document chunks (going INTO the database) and
    "retrieval.query" when embedding the user's question (searching the
    database) — see chat.py. Using the matching task on both sides is what
    makes retrieval accurate.

    This is `async` (uses httpx.AsyncClient, not the sync httpx.post) so
    that waiting on Jina's network round-trip doesn't block FastAPI's event
    loop — without it, one slow embed() call would stall every other
    in-flight request (including anyone else's streaming chat response)
    until it returned.
    """
    async with httpx.AsyncClient() as client:
        response = await client.post(
            JINA_API_URL,
            headers={"Authorization": f"Bearer {settings.jina_api_key}"},
            json={
                "model": JINA_MODEL,
                "task": task,
                "input": texts,
            },
            timeout=60,
        )
    response.raise_for_status()
    data = response.json()["data"]

    # Jina doesn't guarantee response order matches input order — each item
    # carries its original position in "index", so we sort by that to be safe.
    data.sort(key=lambda item: item["index"])
    return [item["embedding"] for item in data]
