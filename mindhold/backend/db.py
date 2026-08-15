"""
Postgres + pgvector — persistent storage for notes and their embeddings.

Two tables, split by purpose:
  - `notes`: what the UI displays (title, description, created_at).
  - `note_chunks`: what retrieval searches over (each note's description,
    split into chunks, each with its own embedding). A short note is
    usually one chunk; a long one becomes several — splitting keeps
    retrieval precise regardless of note length, same reasoning as
    chunking a whole document in ../../06_rag_basics.py.

We use asyncpg directly (no ORM) so every query is plain, visible SQL.
"""

from __future__ import annotations

import asyncpg
from pgvector.asyncpg import register_vector

from embeddings import EMBEDDING_DIM
from settings import settings

_pool: asyncpg.Pool | None = None


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            settings.database_url,
            init=register_vector,  # teaches asyncpg how to send/receive VECTOR values as Python lists
        )
    return _pool


async def init_db() -> None:
    """Create the pgvector extension + tables if they don't exist yet. Safe to call every startup."""
    # The pool's connections auto-register the `vector` type on connect
    # (see get_pool's init=register_vector), which requires the extension
    # to already exist — so CREATE EXTENSION runs first, over a plain
    # one-off connection that skips that registration step.
    conn = await asyncpg.connect(settings.database_url)
    try:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    finally:
        await conn.close()

    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS notes (
                id SERIAL PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)
        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS note_chunks (
                id SERIAL PRIMARY KEY,
                note_id INTEGER NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
                content TEXT NOT NULL,
                embedding VECTOR({EMBEDDING_DIM}) NOT NULL,
                content_tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
            )
        """)
        # Migration for a note_chunks table that already existed before
        # content_tsv was added — CREATE TABLE IF NOT EXISTS above is a
        # no-op against an existing table, so a deployment created before
        # this column existed needs it backfilled explicitly here. Safe to
        # run every startup: the IF NOT EXISTS makes it a no-op once the
        # column is already there.
        await conn.execute("""
            ALTER TABLE note_chunks
            ADD COLUMN IF NOT EXISTS content_tsv TSVECTOR
                GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
        """)
        # GIN index makes the keyword (full-text) search below an index scan
        # instead of a sequential scan over every chunk's tsvector.
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS note_chunks_tsv_idx
            ON note_chunks USING GIN (content_tsv)
        """)


async def create_note(title: str, description: str) -> dict:
    pool = await get_pool()
    async with pool.acquire() as conn:
        record = await conn.fetchrow(
            """
            INSERT INTO notes (title, description)
            VALUES ($1, $2)
            RETURNING id, title, description, created_at
            """,
            title,
            description,
        )
    return dict(record)


async def list_notes() -> list[dict]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        records = await conn.fetch(
            "SELECT id, title, description, created_at FROM notes ORDER BY created_at DESC"
        )
    return [dict(r) for r in records]


async def insert_note_chunks(note_id: int, rows: list[tuple[str, list[float]]]) -> None:
    """rows = list of (chunk_text, embedding_vector) for one note."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.executemany(
            "INSERT INTO note_chunks (note_id, content, embedding) VALUES ($1, $2, $3)",
            [(note_id, content, embedding) for content, embedding in rows],
        )


async def search_similar_chunks(query_embedding: list[float], k: int = 5) -> list[dict]:
    """
    Top-k nearest note chunks to the query embedding, using cosine distance
    (`<=>`, a pgvector operator — smaller distance = more similar). Joins
    back to `notes` so callers get the note's title alongside the matched chunk.

    This is the SEMANTIC half of hybrid search — see hybrid_search.py,
    which calls this alongside a keyword (full-text) search and fuses the
    two ranked lists together.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        records = await conn.fetch(
            """
            SELECT notes.title AS source, note_chunks.content AS content
            FROM note_chunks
            JOIN notes ON notes.id = note_chunks.note_id
            ORDER BY note_chunks.embedding <=> $1
            LIMIT $2
            """,
            query_embedding,
            k,
        )
    return [{"source": r["source"], "content": r["content"]} for r in records]


async def search_keyword_chunks(question: str, k: int = 5) -> list[dict]:
    """
    Top-k note chunks ranked by Postgres full-text search — the KEYWORD half
    of hybrid search, replacing the earlier in-memory BM25 pass. See
    hybrid_search.py, which calls this alongside search_similar_chunks
    (the semantic half) and fuses the two ranked lists together.

    plainto_tsquery() turns the raw question into a tsquery by tokenizing +
    stemming it the same way content_tsv was built (both use the 'english'
    text search config, which matters — a mismatched config means stemmed
    forms don't line up and matches get missed). ts_rank() scores each
    matching row by term frequency, same spirit as BM25's term-frequency
    term, just Postgres's own ranking function instead of a Python library.

    Unlike the old get_all_chunks() (which pulled every chunk into Python
    and ran BM25 there), this pushes the entire keyword search into
    Postgres — the GIN index on content_tsv (see init_db) makes it an
    index scan, not a full table scan, and no chunk text crosses the
    network except the top-k rows actually returned.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        records = await conn.fetch(
            """
            SELECT notes.title AS source, note_chunks.content AS content
            FROM note_chunks
            JOIN notes ON notes.id = note_chunks.note_id
            WHERE note_chunks.content_tsv @@ plainto_tsquery('english', $1)
            ORDER BY ts_rank(note_chunks.content_tsv, plainto_tsquery('english', $1)) DESC
            LIMIT $2
            """,
            question,
            k,
        )
    return [{"source": r["source"], "content": r["content"]} for r in records]
