"""
Hybrid search — Postgres full-text (keyword) search + pgvector (semantic)
search, combined with Reciprocal Rank Fusion (RRF).

WHY HYBRID, NOT JUST SEMANTIC (what chat.py used before this file existed):
Embeddings match on MEANING, not exact words. That's usually great — it's
why "When is Diwali?" can match a note that never uses the word "when" —
but it can miss queries that hinge on an exact keyword, code, or name the
embedding model doesn't weight heavily. Full-text search is the opposite:
strong on exact terms, blind to meaning/paraphrasing. Hybrid search runs
both and merges the results, so a query benefits from whichever approach
handles it better, without you having to guess which one to use per query.

WHY POSTGRES FULL-TEXT SEARCH, NOT BM25-IN-PYTHON (previous version of this
file used the `rank_bm25` library):
BM25 there meant pulling every chunk's text out of the database into
Python on EVERY query, then rebuilding a term-frequency index from scratch
each time — no persistence, and it got slower as the notes table grew.
Postgres's own full-text search (`tsvector`/`tsquery`) does the equivalent
ranking job natively in the database: `note_chunks.content_tsv` is a
generated column (see db.py's init_db) kept in sync automatically whenever
a row is inserted, backed by a GIN index, so a keyword search is an index
scan that returns only the top-k matching rows — no full table pulled over
the network, no re-indexing per request. See db.py's search_keyword_chunks
for the actual query.

RECIPROCAL RANK FUSION (RRF):
Full-text rank scores and cosine-similarity scores are NOT on the same
numeric scale — you can't just average them. RRF sidesteps this by
combining RANKS (1st place, 2nd place, ...) instead of raw scores, since
ranks are comparable no matter what produced them:

    rrf_score += 1 / (k + rank)

k=60 is the standard constant from the original RRF paper — it dampens how
much any single #1 ranking dominates the fused result. A chunk that ranks
decently in BOTH lists ends up scoring higher than a chunk that's #1 in
only one list and absent from the other, which is exactly the "agreement
between two different signals is a strong signal" property that makes
hybrid search better than either method alone.
"""

from __future__ import annotations

import asyncio

from db import search_keyword_chunks, search_similar_chunks
from embeddings import embed


def _reciprocal_rank_fusion(
    ranked_lists: list[list[dict]], k: int = 60, top_n: int = 3
) -> list[dict]:
    """
    Merge multiple ranked lists of chunks into one fused ranking via RRF.
    Chunks are keyed by their text content, since the SAME chunk commonly
    appears in both the keyword and semantic results — that overlap is
    exactly the signal RRF rewards (see module docstring above).
    """
    scores: dict[str, float] = {}
    chunk_by_key: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list):
            key = chunk["content"]
            chunk_by_key[key] = chunk
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)

    fused_keys = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [chunk_by_key[key] for key in fused_keys[:top_n]]


async def hybrid_search(question: str, k: int = 3) -> list[dict]:
    """
    The single entry point chat.py calls. Runs keyword (Postgres full-text)
    and semantic (pgvector) search CONCURRENTLY — both are async I/O calls
    (a DB query and an embedding API call + DB query) with no dependency on
    each other's result, so asyncio.gather runs them at the same time
    instead of one waiting on the other, cutting total latency to roughly
    whichever one is slower rather than the sum of both.
    """
    # Each retrieves more candidates (k=5) than we ultimately want
    # (top_n=k below) — RRF works better with a wider candidate pool to
    # find agreement across, then narrows down to the final top_n at the end.
    async def semantic_search() -> list[dict]:
        [query_vector] = await embed([question], task="retrieval.query")
        return await search_similar_chunks(query_vector, k=5)

    keyword_results, semantic_results = await asyncio.gather(
        search_keyword_chunks(question, k=5),
        semantic_search(),
    )

    return _reciprocal_rank_fusion([keyword_results, semantic_results], top_n=k)
