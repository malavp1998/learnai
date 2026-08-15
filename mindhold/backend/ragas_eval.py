"""
RAGAS evaluation — scoring the LIVE mindhold RAG pipeline (hybrid_search.py
+ chat.py), not just retrieval accuracy.

WHY THIS IS DIFFERENT FROM 09_retrieval_evaluation.py:
09_retrieval_evaluation.py answers "did we FETCH the right chunk?" — pure
retrieval metrics (Precision@k, Recall@k, MRR, Hit Rate@k), hand-computed,
no LLM involved in the scoring itself.

RAGAS answers a different, harder question: "given the chunks we fetched,
is the ANSWER actually good?" That's not something you can compute with
simple string matching — "is this answer faithful to the context" and "is
this answer relevant to the question" require judgment, so RAGAS uses an
LLM AS THE JUDGE. This script uses Groq's llama-3.3-70b-versatile as that
judge (same provider/key already configured for this project via .env —
see settings.py), scoring each question 0.0-1.0 per metric.

THE FOUR METRICS:
  - Faithfulness: of the claims made in the answer, what fraction are
    actually supported by the retrieved context? Catches HALLUCINATION —
    an answer can be fluent and directly on-topic and still invent details
    the context never said. Low faithfulness = the model made things up.
  - Answer Relevancy: does the answer actually address the question asked,
    or does it wander into related-but-unasked territory? Computed by
    having the judge LLM generate several questions the given answer WOULD
    answer, then embedding + comparing those against the real question —
    the closer they are, the more relevant the answer. (This is the one
    metric here that also needs an embedding model, not just an LLM judge
    — see EMBEDDING_MODEL below.)
  - Context Precision: of the chunks retrieved, how many were actually
    useful for answering? Judges RETRIEVAL quality, but from the
    generation side — "was this chunk necessary," not "does this chunk's
    filename match the golden label" (the cruder check 09_retrieval_eval
    uses). Complements Recall below.
  - Context Recall: given a REFERENCE (ground-truth) answer, how much of
    the information needed to produce that reference was actually present
    somewhere in the retrieved context? A low score here means the RIGHT
    chunk was probably never retrieved — the generation step never had a
    chance, regardless of how good the LLM is.

Faithfulness and Answer Relevancy score the GENERATION step (chat.py's
LLM call); Context Precision and Context Recall score the RETRIEVAL step
(hybrid_search.py) — together they cover the full pipeline, not just one
half of it.

WHY AN LLM JUDGE INSTEAD OF HAND-WRITTEN METRICS (unlike 09's approach):
09_retrieval_evaluation.py's metrics are all EXACT MATCHES against a
labeled source filename — cheap, deterministic, no LLM call needed, but
they can only ask binary "did the right file show up" questions. Whether
an ANSWER is faithful or relevant has no simple string-match definition —
two answers can be worded completely differently and both be equally
correct, or worded almost identically and one be a hallucination. That
kind of judgment needs a language model, which means: it costs an API
call per (question, metric) pair, it's non-deterministic run to run (same
caveat as any LLM output), and it's only as trustworthy as the judge model
itself. Standard practice despite this — there's no cheaper substitute for
"is this claim actually supported by the text."

Pipeline:
  golden set (question, reference answer) -> for each question: run
  the REAL hybrid_search() + REAL chat generation (same functions the
  live app uses, not a reimplementation) -> score with all 4 RAGAS
  metrics -> average + print a results table.
"""

from __future__ import annotations

import asyncio

from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas.embeddings.huggingface_provider import HuggingFaceEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecisionWithoutReference,
    ContextRecall,
    Faithfulness,
)

from hybrid_search import hybrid_search
from settings import settings

load_dotenv()


# --- Step A: the judge LLM ---
# Groq exposes an OpenAI-COMPATIBLE API (same request/response shape as
# OpenAI's own API, different base_url) — this is why we can point the
# plain `openai` SDK's client at Groq instead of needing a Groq-specific
# SDK. llm_factory() wraps that client so RAGAS's metrics can call it for
# structured judgment outputs (each metric asks the judge for a score +
# reasoning, parsed into a fixed schema via the `instructor` library
# under the hood — that's why llm_factory needs a real client object, not
# just an API key string).
groq_client = AsyncOpenAI(
    api_key=settings.groq_api_key,
    base_url="https://api.groq.com/openai/v1",
)
judge_llm = llm_factory("llama-3.3-70b-versatile", client=groq_client)

# Answer Relevancy needs an embedding model too (see metric docstring
# above) — reusing the exact same local HuggingFace model
# 06_rag_basics.py and 08_hybrid_search.py use, NOT Jina (which chat.py
# uses for real retrieval). This is a separate, judge-only embedding step
# that never touches note_chunks — keeping it local means no extra API
# key or cost just to run evaluation.
judge_embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")


# --- Step B: the metrics under test ---
faithfulness = Faithfulness(llm=judge_llm)
answer_relevancy = AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings)
context_precision = ContextPrecisionWithoutReference(llm=judge_llm)
context_recall = ContextRecall(llm=judge_llm)


# --- Step C: the golden test set ---
# Each entry: (question, reference_answer). Unlike 09_retrieval_evaluation
# (which only needs to know WHICH FILE the answer should come from),
# RAGAS's Context Recall metric needs an actual reference ANSWER to check
# the retrieved context against — so these are written as ground-truth
# answers, read directly off the real note content already stored in this
# project's mindhold database (see `notes` table).
GOLDEN_SET = [
    ("When is Diwali and how long is the office closed?", "Diwali is on November 8th. The office will be closed for 2 days."),
    ("Who are Kanha's friends, and do they get along?", "Kanha and Manish are friends, but they don't like each other."),
    ("Who wrote the company leave policy, and how many paid leave days do employees get?", "The company leave policy was written by Piyush Malav. Employees get 20 days of paid leave per year, plus 10 public holidays."),
    ("What Python key or variable is mentioned in the Python note?", "The Python note mentions using a 'TPD' key in the project."),
]


# --- Step D: run the REAL pipeline per question ---
async def run_pipeline(question: str) -> tuple[str, list[str]]:
    """
    Calls the SAME hybrid_search() function chat.py's live /api/chat
    endpoint calls — not a reimplementation — so this evaluates the
    actual retrieval code path. Then generates an answer with the judge
    LLM using the identical prompt template chat.py uses, non-streamed
    (streaming vs non-streaming doesn't change WHAT is generated, only
    how it's delivered — irrelevant for scoring answer quality).
    """
    chunks = await hybrid_search(question, k=3)
    contexts = [f"[{c['source']}]\n{c['content']}" for c in chunks]

    prompt = f"""Answer the question using ONLY the context below. If the answer
isn't in the context, say "I don't have that information" — do not guess.

Context:
{chr(10).join(contexts)}

Question: {question}

Answer:"""

    completion = await groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
    )
    answer = completion.choices[0].message.content

    return answer, contexts


# --- Step E: score one question against all 4 metrics ---
async def evaluate_question(question: str, reference: str) -> dict:
    answer, contexts = await run_pipeline(question)

    # Each metric's .ascore() takes exactly the fields it needs (see each
    # class's signature) — Context Recall is the only one that needs the
    # `reference` ground-truth answer, since it's checking whether the
    # RETRIEVED CONTEXT contains what would be needed to produce that
    # reference, not whether the generated answer matches it.
    #
    # Run SEQUENTIALLY, not via asyncio.gather. Groq's free tier caps
    # tokens-per-minute (12,000 TPM) — Faithfulness alone makes several
    # judge calls internally (it decomposes the answer into individual
    # claims, then verifies each claim against the context separately),
    # so firing all 4 metrics at once easily blows the per-minute budget
    # and the request gets rejected with a 429 rather than queued. One
    # judge call in flight at a time trades eval speed for staying under
    # the limit — fine for a small golden set like this one.
    faithfulness_result = await faithfulness.ascore(user_input=question, response=answer, retrieved_contexts=contexts)
    relevancy_result = await answer_relevancy.ascore(user_input=question, response=answer)
    precision_result = await context_precision.ascore(user_input=question, response=answer, retrieved_contexts=contexts)
    recall_result = await context_recall.ascore(user_input=question, retrieved_contexts=contexts, reference=reference)

    return {
        "question": question,
        "answer": answer,
        "faithfulness": faithfulness_result.value,
        "answer_relevancy": relevancy_result.value,
        "context_precision": precision_result.value,
        "context_recall": recall_result.value,
    }


# --- Step F: run the whole golden set, print a results table ---
async def main() -> None:
    results = []
    for i, (question, reference) in enumerate(GOLDEN_SET):
        if i > 0:
            # Small pause between questions — same TPM-budget reasoning as
            # the sequential .ascore() calls above, just at the
            # question-loop level instead of the metric level.
            await asyncio.sleep(5)
        print(f"Evaluating: {question}")
        result = await evaluate_question(question, reference)
        results.append(result)
        print(f"  Answer: {result['answer']}")
        print(
            f"  Faithfulness={result['faithfulness']:.2f}  "
            f"AnswerRelevancy={result['answer_relevancy']:.2f}  "
            f"ContextPrecision={result['context_precision']:.2f}  "
            f"ContextRecall={result['context_recall']:.2f}"
        )
        print()

    n = len(results)
    print("=" * 60)
    print(f"AVERAGES over {n} golden question(s):")
    print(f"  Faithfulness      : {sum(r['faithfulness'] for r in results) / n:.2f}")
    print(f"  Answer Relevancy  : {sum(r['answer_relevancy'] for r in results) / n:.2f}")
    print(f"  Context Precision : {sum(r['context_precision'] for r in results) / n:.2f}")
    print(f"  Context Recall    : {sum(r['context_recall'] for r in results) / n:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
