"""
FastAPI app entrypoint.

Run:  uvicorn main:app --reload --port 8000

Endpoints:
  POST /api/notes  — create a note (title + description); the description
                      is chunked + embedded immediately so it becomes
                      searchable by the chat endpoint right away.
  GET  /api/notes   — list all notes, newest first, for the Notes grid.
  POST /api/chat    — ask a question, get an answer streamed back via SSE,
                      retrieved from the notes stored above.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel

from chat import stream_chat_response
from db import create_note, init_db, insert_note_chunks, list_notes
from embeddings import embed

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(lifespan=lifespan)

# The React dev server (Vite, default port 5173) runs on a different origin
# than this API (port 8000) — browsers block cross-origin requests unless
# the server explicitly allows it via CORS headers. This is a browser-only
# rule (curl/Postman ignore it), which is why curl testing works before the
# frontend does.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str


class NoteCreateRequest(BaseModel):
    title: str
    description: str


@app.post("/api/notes")
async def add_note(request: NoteCreateRequest):
    note = await create_note(request.title, request.description)

    # Same split -> embed -> store pattern the old ingest.py used for
    # docs/*.txt, just triggered per-note instead of as a batch script.
    chunks = splitter.split_text(request.description)
    vectors = await embed(chunks, task="retrieval.passage")
    await insert_note_chunks(note["id"], list(zip(chunks, vectors)))

    return note


@app.get("/api/notes")
async def get_notes():
    return await list_notes()


@app.post("/api/chat")
async def chat(request: ChatRequest):
    return StreamingResponse(
        stream_chat_response(request.question),
        media_type="text/event-stream",
    )
