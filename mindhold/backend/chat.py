"""
The streaming RAG endpoint — this is the piece that answers your question
of "how does the UI see the response in batches while the LLM keeps
generating?"

The short answer: there ARE no batches on the backend. The LLM generates
one token at a time. As each token arrives from Groq, we immediately
`yield` it onward to the browser as one Server-Sent Event (SSE). Nothing
here waits or buffers. What looks like "batches" in the UI is really just:
tokens arriving in quick, irregular bursts, and React re-rendering each
time a new piece of text lands — see frontend/src/useChatStream.js for the
receiving side of this exact same stream.

Compare to 06_rag_basics.py's `ask()`: same retrieve -> prompt -> LLM call,
but there `chain.invoke()` blocks until the ENTIRE answer is ready, then
prints it all at once. Here we use `.astream()` instead of `.invoke()` —
that's the one-line difference that turns "wait then get it all" into
"get pieces as they're produced".
"""

from __future__ import annotations

import json

from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq

from hybrid_search import hybrid_search
from settings import settings

llm = ChatGroq(model="openai/gpt-oss-20b", api_key=settings.groq_api_key)

PROMPT_TEMPLATE = """Answer the question using ONLY the context below. If the answer
isn't in the context, say "I don't have that information" — do not guess.

Context:
{context}

Question: {question}

Answer:"""


def format_context(chunks: list[dict]) -> str:
    return "\n\n".join(f"[{c['source']}]\n{c['content']}" for c in chunks)


async def stream_chat_response(question: str):
    """
    An async generator: each `yield` produces one SSE "event" line, sent to
    the browser the instant it's produced. FastAPI's StreamingResponse
    pulls from this generator and writes each piece straight to the open
    HTTP connection — no waiting for the generator to finish.
    """
    # Step 1+2: hybrid retrieval — runs Postgres full-text (keyword) search
    # and pgvector (semantic) search over note_chunks CONCURRENTLY, then
    # fuses the two ranked lists with Reciprocal Rank Fusion. See
    # hybrid_search.py for the full explanation of why hybrid beats either
    # search alone, and how RRF merges two differently-scaled ranking
    # systems into one.
    chunks = await hybrid_search(question, k=3)

    sources = [c["source"] for c in chunks]
    # First SSE event: tell the frontend which sources were retrieved,
    # before any answer text exists yet — lets the UI show "searching..."
    # then immediately show sources while the answer is still streaming in.
    yield f"event: sources\ndata: {json.dumps(sources)}\n\n"

    prompt = PROMPT_TEMPLATE.format(context=format_context(chunks), question=question)

    # Step 3: stream the LLM's answer. `.astream()` yields chunks as the
    # model generates them (as opposed to `.invoke()`, which blocks until
    # the full answer is ready). Each chunk here is typically a few
    # characters to a few words, decided by Groq's inference engine — not by us.
    async for chunk in llm.astream([SystemMessage(content=prompt)]):
        if chunk.content:
            # SSE format: "data: <payload>\n\n" — the blank line is what
            # marks the end of one event. json.dumps escapes newlines so a
            # multi-line chunk can't break the one-event-per-line SSE format.
            yield f"event: token\ndata: {json.dumps(chunk.content)}\n\n"

    yield "event: done\ndata: {}\n\n"
