"""
STEP 11: Query transformation — rewriting, decomposition, and HyDE.

WHERE THIS FITS: 08_hybrid_search.py improved WHICH RETRIEVERS run.
10_reranking.py improved HOW CANDIDATES ARE RE-SCORED after retrieval.
Both of those assume the QUERY ITSELF is a reasonable thing to search
with. Often it isn't — a real user's question can be vague, typo'd,
phrased as two questions stitched together, or worded nothing like the
document that actually answers it. No amount of better retrieval or
reranking fixes a bad query; you have to fix the query BEFORE it ever
reaches retrieval. That's what every technique in this file does.

THREE TECHNIQUES, THREE DIFFERENT QUERY PROBLEMS:

  1. QUERY REWRITING — problem: the raw question is vague, colloquial,
     or missing context a search index needs. Fix: ask an LLM to rewrite
     it into a cleaner, more explicit search query BEFORE retrieval runs.
     Example: "how much is the cheap one" -> "What is the monthly price
     of the CloudSync Starter plan?"

  2. MULTI-QUERY DECOMPOSITION — problem: the question is actually
     SEVERAL questions bundled into one ("compare X and Y", "what's the
     price AND how many devices does it support"). A single retrieval
     pass optimizes for ONE topic and may starve the other. Fix: ask an
     LLM to split the question into independent sub-questions, retrieve
     for EACH separately, then merge/dedupe the results.

  3. HyDE (Hypothetical Document Embeddings) — problem: for SEMANTIC
     search specifically, a question and its answer are often worded
     very differently ("How do I stay healthy?" vs. a document written
     as declarative facts: "A balanced diet consists of..."). Bi-encoder
     embeddings work by proximity in vector space, and a question-shaped
     sentence doesn't always land close to an answer-shaped sentence
     even when the answer is correct. Fix: ask an LLM to write a FAKE,
     plausible-sounding ANSWER to the question first (it doesn't need to
     be factually correct — it just needs to be answer-SHAPED), then
     embed THAT instead of the raw question. An invented answer is
     stylistically much closer to a real document chunk than a question
     is, so it retrieves better.

ALL THREE SHARE ONE COST: each adds at least one extra LLM call BEFORE
retrieval even starts — rewriting is +1 call, decomposition is +1 call
(to split) + one retrieval pass per sub-question, HyDE is +1 call (to
generate the hypothetical answer). This is a real latency/cost tradeoff,
not a free upgrade — you're spending an LLM call to make the SEARCH
better, which only pays off when query quality is actually the
bottleneck (as opposed to retrieval or reranking quality, which
08_hybrid_search.py and 10_reranking.py already address).

Pipeline for this script (steps A-E reuse 10_reranking.py's full stack —
hybrid retrieval + RRF + cross-encoder reranking — unchanged): load docs
-> split into chunks -> build BM25 + FAISS + reranker -> [NEW] transform
the query one of three ways -> run the SAME retrieval+rerank pipeline on
the transformed query/queries -> stuff into a prompt -> ask the LLM.
"""

from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder

load_dotenv()

DOCS_DIR = Path(__file__).parent / "docs"


# --- Step A-D: identical setup to 10_reranking.py (load, split, index, reranker) ---
documents = []
for file_path in sorted(DOCS_DIR.glob("*.txt")):
    documents.extend(TextLoader(str(file_path)).load())

print(f"Loaded {len(documents)} document(s) from {DOCS_DIR}")

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

print(f"Split into {len(chunks)} chunk(s)\n")

CANDIDATE_POOL_SIZE = 8
FINAL_TOP_N = 3

bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = CANDIDATE_POOL_SIZE

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": CANDIDATE_POOL_SIZE})

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

llm = ChatGroq(model="openai/gpt-oss-20b")


def reciprocal_rank_fusion(ranked_lists: list[list], k: int = 60, top_n: int = CANDIDATE_POOL_SIZE) -> list:
    """Same RRF logic as 08_hybrid_search.py / 10_reranking.py."""
    scores: dict[str, float] = {}
    doc_by_key: dict[str, object] = {}
    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list):
            key = doc.page_content
            doc_by_key[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    fused_keys = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [doc_by_key[key] for key in fused_keys[:top_n]]


def rerank(query_for_ranking: str, candidates: list, top_n: int = FINAL_TOP_N) -> list:
    """Same reranking logic as 10_reranking.py. `query_for_ranking` is
    deliberately a separate arg from whatever query was used to RETRIEVE
    the candidates — see hyde_search() below, where these differ."""
    if not candidates:
        return []
    pairs = [(query_for_ranking, doc.page_content) for doc in candidates]
    scores = reranker.predict(pairs)
    scored = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
    return [doc for doc, _score in scored[:top_n]]


def retrieve_and_rerank(search_query: str, rank_query: str) -> list:
    """The full 10_reranking.py pipeline as one reusable function: hybrid
    retrieval on `search_query`, then rerank against `rank_query`. Almost
    always these are the SAME string — they only diverge for HyDE, where
    you SEARCH with a fake hypothetical answer but still want to RANK
    candidates against the user's real question (see hyde_search)."""
    bm25_results = bm25_retriever.invoke(search_query)
    semantic_results = semantic_retriever.invoke(search_query)
    candidate_pool = reciprocal_rank_fusion([bm25_results, semantic_results])
    return rerank(rank_query, candidate_pool)


ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """Answer the question using ONLY the context below. If the answer
isn't in the context, say "I don't have that information" — do not guess.

Context:
{context}

Question: {question}

Answer:"""
)


def format_docs(docs):
    return "\n\n".join(f"[{Path(d.metadata['source']).name}]\n{d.page_content}" for d in docs)


def generate_answer(context_docs: list, question: str) -> str:
    chain = ANSWER_PROMPT | llm | StrOutputParser()
    return chain.invoke({"context": format_docs(context_docs), "question": question})


# ============================================================
# Technique 1: QUERY REWRITING
# ============================================================
# Rewrites a vague/colloquial question into an explicit search query
# BEFORE retrieval runs. This is a single, cheap LLM call — the rewrite
# prompt is deliberately narrow (just "produce a better search query",
# nothing else) so the model doesn't wander into answering the question
# itself or adding commentary.
REWRITE_PROMPT = ChatPromptTemplate.from_template(
    """Rewrite the user's question into a clear, explicit search query
suitable for a document search engine. Expand vague references, spell
out what's being asked, and use likely terminology from the source
material — but do not answer the question, only rewrite it.

Respond with ONLY the rewritten query, nothing else.

User's question: {question}

Rewritten query:"""
)


def rewrite_query(question: str) -> str:
    chain = REWRITE_PROMPT | llm | StrOutputParser()
    return chain.invoke({"question": question}).strip()


def ask_with_rewriting(question: str):
    rewritten = rewrite_query(question)
    print(f"Q (original):  {question}")
    print(f"Q (rewritten): {rewritten}")

    final_chunks = retrieve_and_rerank(search_query=rewritten, rank_query=rewritten)
    print("Retrieved:     ", [Path(d.metadata["source"]).name for d in final_chunks])

    answer = generate_answer(final_chunks, question)  # answer the ORIGINAL question
    print(f"A: {answer}\n")
    print("-" * 60)


# ============================================================
# Technique 2: MULTI-QUERY DECOMPOSITION
# ============================================================
# Splits a compound question into independent sub-questions, retrieves
# for EACH separately, then merges the results. A single retrieval pass
# over a compound question tends to skew toward whichever half of the
# question happens to dominate the embedding/keyword signal — splitting
# guarantees each sub-topic gets its own dedicated retrieval pass.
DECOMPOSE_PROMPT = ChatPromptTemplate.from_template(
    """Break the user's question into the smallest number of independent
sub-questions needed to fully answer it. If the question is already a
single, simple question, return it unchanged as the only sub-question.

Respond with ONLY the sub-questions, one per line, no numbering, no
other text.

User's question: {question}

Sub-questions:"""
)


def decompose_query(question: str) -> list[str]:
    chain = DECOMPOSE_PROMPT | llm | StrOutputParser()
    raw = chain.invoke({"question": question})
    # One sub-question per line, dropping any blank lines the model
    # might produce.
    return [line.strip() for line in raw.splitlines() if line.strip()]


def ask_with_decomposition(question: str):
    sub_questions = decompose_query(question)
    print(f"Q (original): {question}")
    print(f"Sub-questions ({len(sub_questions)}):")
    for sq in sub_questions:
        print(f"  - {sq}")

    # Retrieve for EACH sub-question independently, then merge into one
    # pool via RRF again — a second, higher-level fusion, this time
    # fusing across SUB-QUESTIONS instead of across RETRIEVERS. Reuses
    # the exact same reciprocal_rank_fusion() function for both purposes,
    # since RRF only cares that its inputs are ranked lists of the same
    # kind of item — it doesn't care WHY each list was produced.
    per_subquestion_results = [
        rerank(sq, reciprocal_rank_fusion([bm25_retriever.invoke(sq), semantic_retriever.invoke(sq)]))
        for sq in sub_questions
    ]
    merged_chunks = reciprocal_rank_fusion(per_subquestion_results, top_n=FINAL_TOP_N)

    print("Merged retrieval:", [Path(d.metadata["source"]).name for d in merged_chunks])

    answer = generate_answer(merged_chunks, question)  # answer the ORIGINAL compound question
    print(f"A: {answer}\n")
    print("-" * 60)


# ============================================================
# Technique 3: HyDE (Hypothetical Document Embeddings)
# ============================================================
# Generates a FAKE, plausible-sounding answer to the question, then uses
# THAT (not the raw question) as the text to embed for semantic search.
# The hypothetical answer doesn't need to be factually correct — its
# only job is to be worded like a real document passage would be, since
# that's what makes it land close to real document chunks in embedding
# space. BM25 (keyword search) still runs on the ORIGINAL question,
# since HyDE is specifically a fix for semantic/embedding search's
# question-vs-answer phrasing mismatch, not a general search-quality fix.
HYDE_PROMPT = ChatPromptTemplate.from_template(
    """Write a short, plausible-sounding passage that WOULD answer the
question below, as if it were an excerpt from a reference document. It
does not need to be factually correct — focus on matching the STYLE and
PHRASING of a real informational document, not accuracy.

Question: {question}

Hypothetical passage:"""
)


def generate_hypothetical_answer(question: str) -> str:
    chain = HYDE_PROMPT | llm | StrOutputParser()
    return chain.invoke({"question": question}).strip()


def ask_with_hyde(question: str):
    hypothetical = generate_hypothetical_answer(question)
    print(f"Q (original):          {question}")
    print(f"Hypothetical passage:  {hypothetical}")

    # BM25 still searches with the REAL question (keyword search doesn't
    # have the question-vs-answer phrasing problem HyDE solves). Semantic
    # search embeds the HYPOTHETICAL passage instead of the question.
    bm25_results = bm25_retriever.invoke(question)
    semantic_results = semantic_retriever.invoke(hypothetical)
    candidate_pool = reciprocal_rank_fusion([bm25_results, semantic_results])

    # Reranking always compares against the REAL question, never the
    # hypothetical passage — the cross-encoder's job is to judge whether
    # a candidate actually answers what the USER asked, and the
    # hypothetical passage was only ever a retrieval trick, not something
    # we want influencing the final relevance judgment.
    final_chunks = rerank(question, candidate_pool)

    print("Retrieved:             ", [Path(d.metadata["source"]).name for d in final_chunks])

    answer = generate_answer(final_chunks, question)
    print(f"A: {answer}\n")
    print("-" * 60)


if __name__ == "__main__":
    print("=" * 60)
    print("TECHNIQUE 1: Query Rewriting")
    print("=" * 60)
    # Deliberately vague/colloquial — a raw retrieval pass on this exact
    # wording has little to latch onto; the rewrite step should turn it
    # into something that names the actual concept (pricing plan, storage).
    ask_with_rewriting("how much is the cheap one and what do you get")

    print("\n" + "=" * 60)
    print("TECHNIQUE 2: Multi-Query Decomposition")
    print("=" * 60)
    # A genuinely compound question spanning TWO different source docs —
    # a single retrieval pass tends to favor whichever topic dominates
    # the embedding, starving the other.
    ask_with_decomposition(
        "How many days of paid leave do employees get, and how much "
        "does the CloudSync Pro plan cost per month?"
    )

    print("\n" + "=" * 60)
    print("TECHNIQUE 3: HyDE (Hypothetical Document Embeddings)")
    print("=" * 60)
    # Phrased as a personal, conversational question with none of the
    # source document's vocabulary ("balanced diet", "macronutrients") —
    # a case where a fake answer-shaped passage should land closer to
    # the real document than the question itself would.
    ask_with_hyde("What should I eat if I want to be healthier?")
