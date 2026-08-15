# MindHold Backend

A FastAPI backend for a personal notes + RAG chat app. You write notes; the
app makes them searchable and lets you ask questions about them in natural
language, with the answer generated only from what your notes actually say
(not the LLM's general knowledge) and streamed back token-by-token.

This is the production-shaped evolution of the standalone learning scripts
at the repo root (`../../06_rag_basics.py`, `../../08_hybrid_search.py`) —
same core RAG ideas, but backed by a real persistent database, a real API
server, and a real (measured, not assumed) evaluation of answer quality.

---

## What this project does

1. **Store notes.** `POST /api/notes` saves a title + description. The
   description is immediately split into chunks and embedded, so it becomes
   searchable right away — no separate ingestion step to remember to run.
2. **Answer questions about your notes.** `POST /api/chat` takes a question,
   retrieves the most relevant chunks across all your notes, and asks an LLM
   to answer **using only that retrieved content** — this is Retrieval-
   Augmented Generation (RAG). If the answer isn't in your notes, the model
   is instructed to say so instead of guessing.
3. **Stream the answer.** The response comes back as Server-Sent Events
   (SSE) — tokens appear as the LLM generates them, not all at once after a
   long wait.
4. **Retrieve with both keyword AND meaning matching.** Retrieval isn't just
   "closest embedding" — it combines Postgres full-text (keyword) search
   with pgvector semantic search, merged with Reciprocal Rank Fusion, so an
   exact term/name and a paraphrased question are both handled well.
5. **Measure whether any of this actually works.** A RAGAS evaluation script
   scores real answers from the real pipeline against a hand-written golden
   question set — see [Evaluation Results](#evaluation-results) below.

---

## Tech stack

| Concern | Technology | Why |
|---|---|---|
| API server | **FastAPI** + **uvicorn** | Async-native, so a slow embedding/DB call doesn't block other in-flight requests (see `embeddings.py`) |
| Database | **Postgres** + **pgvector** extension (`pgvector/pgvector:pg16` Docker image) | One database for both normal relational data (notes) and vector similarity search — no separate vector-DB service to run |
| DB driver | **asyncpg** (no ORM) | Every query is plain, visible SQL — deliberate for a learning project; nothing hidden behind an ORM's query builder |
| Keyword search | Postgres **full-text search** (`tsvector`/`tsquery`, `ts_rank`, GIN index) | Native to the database already in use — no extra library, no in-memory index rebuilt on every query (see [Why Postgres full-text, not BM25](#why-postgres-full-text-search-not-bm25)) |
| Semantic search | **pgvector** cosine distance (`<=>` operator) | Vector similarity search as a normal SQL `ORDER BY`, in the same database as everything else |
| Embeddings | **Jina AI** (`jina-embeddings-v3`) via HTTP API | API-based, not a local model — no download, no local compute cost; see `embeddings.py` |
| LLM (answers) | **Groq** (`openai/gpt-oss-20b`) via `langchain-groq` | Fast inference, used for the actual chat answers |
| LLM (RAGAS judge) | **Groq** (`llama-3.3-70b-versatile`) via the `openai` SDK pointed at Groq's OpenAI-compatible endpoint | Same provider/API key as the rest of the app — no second API key needed just for evaluation |
| Config | **pydantic-settings** | Validates all required env vars once at startup; fails fast with a clear message instead of a cryptic `KeyError` deep inside a request |
| Evaluation | **RAGAS** | Standard framework for scoring RAG answer quality (faithfulness, relevancy) and retrieval quality (context precision/recall) — see [Evaluation Results](#evaluation-results) |
| Chunking | **langchain-text-splitters** (`RecursiveCharacterTextSplitter`) | Same 500-char/50-overlap chunking used throughout this repo's learning scripts |

---

## Project structure

```
mindhold/
├── .env                      # your local secrets (GROQ_API_KEY, JINA_API_KEY, DATABASE_URL) — not committed
├── .env.example               # template for the above
├── docker-compose.yml         # Postgres+pgvector container definition
├── backend/                   # <- this directory
│   ├── main.py                 # FastAPI app + routes (POST /api/notes, GET /api/notes, POST /api/chat)
│   ├── settings.py             # pydantic-settings config, reads .env
│   ├── db.py                   # asyncpg pool, schema (notes, note_chunks), all SQL queries
│   ├── embeddings.py           # Jina AI embedding API client
│   ├── hybrid_search.py        # keyword + semantic retrieval, fused via Reciprocal Rank Fusion
│   ├── chat.py                 # RAG prompt + streaming LLM call (SSE generator)
│   ├── ragas_eval.py            # RAGAS evaluation harness — see below
│   ├── requirements.txt
│   └── venv/                   # local virtualenv (not committed)
└── frontend/                  # React + Vite UI (Notes tab, Chat tab) — see frontend/README.md
```

---

## How it works, end to end

### 1. Creating a note (`POST /api/notes`)

```
title + description
        │
        ▼
  INSERT INTO notes         (the row the UI displays)
        │
        ▼
  split description into ~500-char chunks (RecursiveCharacterTextSplitter)
        │
        ▼
  embed each chunk via Jina AI (task="retrieval.passage")
        │
        ▼
  INSERT INTO note_chunks   (content + embedding, one row per chunk)
```

A short note is usually one chunk; a long one becomes several. Splitting
`notes` (display data) from `note_chunks` (retrieval data) means a long
note becomes independently searchable per-section without changing the
schema — see `db.py`'s module docstring.

### 2. Asking a question (`POST /api/chat`)

```
question
   │
   ├─────────────────────────────┬─────────────────────────────┐
   ▼ (concurrently, asyncio.gather)                             ▼
Postgres full-text search                              embed question (Jina)
(ts_rank + plainto_tsquery,                                      │
 GIN-indexed, top 5)                                              ▼
   │                                                    pgvector cosine search
   │                                                    (top 5)
   └──────────────────┬───────────────────────────────────────┘
                       ▼
         Reciprocal Rank Fusion (merge both ranked lists by RANK,
         not raw score — the two scoring systems aren't comparable)
                       ▼
              top 3 fused chunks
                       ▼
     prompt: "answer using ONLY this context" + question
                       ▼
        Groq LLM (openai/gpt-oss-20b), streamed token by token
                       ▼
     SSE events → browser: `sources` → `token` × N → `done`
```

### Why hybrid search, not semantic search alone?

Embeddings match on **meaning**, not exact words — great for a question
like "When is Diwali?" matching a note that never uses the word "when," but
weak for a question that hinges on an exact keyword, code, or name the
embedding model doesn't weight heavily. Keyword search is the opposite:
strong on exact terms, blind to paraphrasing. Running both and merging the
results means a query benefits from whichever approach actually handles it
better, without the app having to guess which one to use per question.

### Why Postgres full-text search, not BM25?

An earlier version of `hybrid_search.py` used the `rank_bm25` Python
library — but that meant pulling every chunk's text out of the database
into Python **on every single query**, then rebuilding a term-frequency
index from scratch each time. No persistence, and it got slower as the
notes table grew.

Postgres's own full-text search does the equivalent ranking job natively in
the database: `note_chunks.content_tsv` is a `GENERATED ALWAYS AS ... STORED`
column (see `db.py`'s `init_db`), automatically kept in sync on every
insert, backed by a **GIN index**. A keyword search is then a plain indexed
`SELECT ... WHERE content_tsv @@ plainto_tsquery(...) ORDER BY ts_rank(...)`
— an index scan, not a full table scan, and no chunk text crosses the
network except the top-k rows actually returned.

### Reciprocal Rank Fusion (RRF)

Full-text `ts_rank` scores and pgvector cosine-distance scores are **not on
the same numeric scale** — you can't just average them. RRF sidesteps this
by combining **ranks** (1st place, 2nd place, ...) instead of raw scores,
since rank position means the same thing regardless of which system
produced it:

```
rrf_score += 1 / (k + rank)
```

`k=60` is the standard constant from the original RRF paper — it dampens
how much any single #1 ranking dominates the fused result. A chunk that
ranks decently in **both** lists ends up scoring higher than a chunk that's
#1 in only one list and absent from the other — agreement between two
different search strategies is treated as a strong signal.

### Why streaming?

`chat.py` uses `llm.astream(...)` instead of `llm.invoke(...)` — the LLM
generates one token at a time internally, and each token is `yield`-ed as
one SSE event the instant it's produced. There's no batching or buffering
on the backend; what looks like "batches" arriving in the UI is really just
network delivery timing plus React re-rendering on each new piece of text.
See `frontend/src/useChatStream.js` for the receiving side.

---

## Setup

```bash
# 1. Start Postgres (with pgvector) — from the mindhold/ directory
cd ..
docker compose up -d

# 2. Configure secrets
cp .env.example .env
# then fill in GROQ_API_KEY (https://console.groq.com/keys)
# and JINA_API_KEY (https://jina.ai/embeddings)
# DATABASE_URL is pre-filled to match docker-compose.yml — no change needed

# 3. Install backend dependencies
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 4. Run the API server
uvicorn main:app --reload --port 8000
```

The server creates the `notes`/`note_chunks` tables (and the `content_tsv`
generated column + GIN index) automatically on startup — no separate
migration step, safe to run repeatedly.

**Frontend:** see `../frontend/README.md` — `npm install && npm run dev`,
runs on `http://localhost:5173` and talks to this backend on port 8000.

**Troubleshooting a connection-refused error:** run `docker ps` and check
the `PORTS` column for `mindhold_postgres` shows `0.0.0.0:5432->5432/tcp`,
not bare `5432/tcp`. If it's the latter (port not actually published to the
host), run `docker compose up -d --force-recreate` from `mindhold/` — this
rebuilds the container from `docker-compose.yml`'s config without touching
the data volume (`pgdata`), so nothing stored is lost.

---

## API endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/notes` | Create a note. Body: `{"title": str, "description": str}`. Chunks + embeds the description immediately. |
| `GET` | `/api/notes` | List all notes, newest first. |
| `POST` | `/api/chat` | Ask a question. Body: `{"question": str}`. Returns an SSE stream (`event: sources`, then `event: token` × N, then `event: done`). |

---

## Evaluation

### Two different eval scripts, two different questions

| Script | Question it answers | How |
|---|---|---|
| `../../09_retrieval_evaluation.py` | *Did retrieval fetch the right chunk?* | Hand-computed Precision@k / Recall@k / MRR / Hit Rate@k against a golden (question → expected source file) set. No LLM involved in scoring — pure, deterministic, cheap. Runs against the standalone `08_hybrid_search.py` (BM25 + FAISS), not this live app. |
| `ragas_eval.py` (this directory) | *Given what was retrieved, is the generated ANSWER actually good — grounded, relevant, and did retrieval find what generation needed?* | **RAGAS**, using an LLM as judge. Runs the **real, live** pipeline — the actual `hybrid_search()` function and the actual chat-generation prompt this app uses, not a reimplementation. |

`ragas_eval.py` is the one that matters for "is this RAG app actually good,"
because retrieving the technically-correct chunk doesn't guarantee the LLM
writes a faithful, relevant answer from it — that gap is exactly what
hand-computed retrieval metrics can't see and RAGAS is built to measure.

### The four RAGAS metrics

- **Faithfulness** — of the claims made in the generated answer, what
  fraction are actually supported by the retrieved context? Catches
  **hallucination**: an answer can be fluent and on-topic and still invent
  a detail the notes never said.
- **Answer Relevancy** — does the answer actually address the question
  asked, or wander into related-but-unasked territory? (Also needs an
  embedding model — a local `sentence-transformers` model is used here,
  judge-only, never touching real note embeddings.)
- **Context Precision** — of the chunks retrieved, how many were actually
  useful for answering? Scores retrieval quality from the generation side.
- **Context Recall** — given a known-correct reference answer, how much of
  the information needed to produce it was present *somewhere* in the
  retrieved context? A low score here means the right chunk was probably
  never retrieved at all — generation never had a chance, regardless of how
  good the LLM is.

Faithfulness and Answer Relevancy score the **generation** step (`chat.py`);
Context Precision and Context Recall score the **retrieval** step
(`hybrid_search.py`) — together, all four cover the full pipeline.

### Why an LLM judge instead of exact-match metrics

Two answers can be worded completely differently and both be equally
correct, or worded nearly identically and one be a hallucination — there's
no simple string-match definition of "faithful" or "relevant." That
judgment needs a language model, which means each metric costs an API call
per question, is somewhat non-deterministic run to run (like any LLM
output), and is only as trustworthy as the judge model itself. This is
standard practice in RAG evaluation regardless — there's no cheaper
substitute for "is this claim actually supported by the text."

### The golden set

Four questions, each with a hand-written reference answer, written directly
against real notes already stored in this project's database (Diwali
holiday note, a friends note, the company leave policy note, a Python note)
— not fabricated scenarios. See `GOLDEN_SET` in `ragas_eval.py`.

### Running it

```bash
cd backend
source venv/bin/activate
python ragas_eval.py
```

Requires: Postgres running with real notes in it (see Setup above),
`GROQ_API_KEY` set. Metrics run **sequentially**, with a short pause
between questions — Groq's free tier caps tokens-per-minute, and
Faithfulness alone makes several internal judge calls per question
(it decomposes the answer into individual claims and verifies each one
separately), so running everything concurrently reliably triggers a 429
rate-limit error. Expect the full run to take a couple of minutes.

---

## Evaluation Results

Last measured run, 4 golden questions, `llama-3.3-70b-versatile` as judge:

| Metric | Average | What it means here |
|---|---|---|
| **Faithfulness** | **0.94** | Answers are almost entirely grounded in retrieved context — one question showed a partial hallucination (see below) |
| **Answer Relevancy** | **0.76** | Generally on-topic; pulled down by one overly terse answer (see below) |
| **Context Precision** | **1.00** | Every retrieved chunk was actually useful — no wasted/irrelevant retrievals |
| **Context Recall** | **1.00** | The right information was found in retrieval every single time |

**Reading these together:** retrieval (hybrid search) is performing
perfectly against this golden set — Context Precision and Recall both at
1.00 mean the RRF-fused hybrid search is consistently finding exactly what's
needed. The weaker scores are entirely on the **generation** side (the LLM
call in `chat.py`), not retrieval — a useful distinction, since it means if
answer quality needs improving, the fix is in the prompt/generation step,
not in the search/indexing logic.

**Per-question detail from the last run:**

| Question | Faithfulness | Answer Relevancy | Notes |
|---|---|---|---|
| "When is Diwali and how long is the office closed?" | 1.00 | 0.96 | Clean |
| "Who are Kanha's friends, and do they get along?" | 0.75 | 0.94 | Answer added "they are both friends with the speaker" — a claim not clearly supported by the source note. A real, caught hallucination. |
| "Who wrote the company leave policy, and how many paid leave days do employees get?" | 1.00 | 0.97 | Clean |
| "What Python key or variable is mentioned in the Python note?" | 1.00 | 0.17 | Answer was just "TPD" — technically correct but so terse the judge scored it as barely addressing the question as phrased |

Context Precision and Context Recall were 1.00 on every individual question
in this run, not just on average.

**What this demonstrates in practice:** RAGAS caught two concrete, real
issues in this app's live answers that simple retrieval metrics (like
`09_retrieval_evaluation.py`'s Precision@k/Recall@k) cannot see, because
those only check "did we fetch the right file" — they have no way to
evaluate whether the LLM's *answer* stayed faithful to what it was given,
or whether the answer was actually a satisfying response to the question.

*Note: these numbers come from a small (4-question), hand-written golden
set built from personal dev-database notes — useful for catching gross
regressions and demonstrating the eval pipeline, but not a statistically
rigorous benchmark. A production deployment would want a larger, more
diverse golden set and a fixed/versioned dataset rather than live personal
data.*

---

## What's implemented vs. what's next

**Implemented:**
- Note storage with automatic chunking + embedding on creation
- Hybrid retrieval (Postgres full-text + pgvector semantic), fused with RRF
- Streaming RAG chat via SSE
- RAGAS-based generation + retrieval quality evaluation

**Not yet implemented** (see `../../NOTES.md` for the full project-wide
roadmap, which covers both this app and the standalone learning scripts):
- Reranking (a cross-encoder re-scoring the fused top-k before it reaches the LLM)
- Structured output (reliable JSON/Pydantic-typed answers)
- A larger, versioned evaluation golden set independent of live personal data
- Conversation memory in the chat endpoint (each `/api/chat` call is currently stateless — no multi-turn history)
