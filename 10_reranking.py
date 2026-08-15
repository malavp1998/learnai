"""
STEP 10: Reranking — a cross-encoder re-scores the hybrid-search
candidates before they reach the LLM.

WHERE THIS FITS: 08_hybrid_search.py already fixed one problem — pure
semantic search misses exact keywords, pure BM25 misses paraphrasing, so
hybrid + RRF fuses both into one ranked list. This file adds a SECOND,
different fix, on top of that: even RRF's fused ranking is only as good
as the two retrievers feeding it, and neither of them actually reads the
query and the document TOGETHER. Reranking closes that gap.

BI-ENCODER vs CROSS-ENCODER — the core distinction:

  BI-ENCODER (what 08_hybrid_search.py's semantic retriever uses):
  The query gets embedded into a vector. Each document chunk was ALREADY
  embedded into a vector, separately, ahead of time (that's what makes
  it fast — chunk embeddings are precomputed once and stored in FAISS/
  pgvector). At search time you just compare the query vector to
  millions of precomputed chunk vectors with cheap math (cosine
  distance). The model never sees the query and a specific document
  side by side — it only ever sees them one at a time, in isolation.

  CROSS-ENCODER (what THIS file adds):
  Takes the query and ONE document TOGETHER, as a single input:
  "[query] [SEP] [document]" — and the model reads both at once, letting
  it directly attend between every word of the query and every word of
  the document. This produces a much more accurate relevance score,
  because the model can reason about how they relate to each other
  specifically, not just "are these two independently-computed points
  close together in vector space."

WHY NOT USE A CROSS-ENCODER FOR EVERYTHING, THEN?
Cost. A cross-encoder has to re-run the full model for EVERY
(query, document) PAIR, live, at query time — there's no precomputing
"the query's meaning" ahead of time, because the model needs both texts
present together. Scoring 3 chunks is fine. Scoring 10,000 chunks against
every query would be far too slow for a production app.

THE STANDARD PATTERN — cheap+wide, then expensive+narrow:
  1. Use fast, cheap retrieval (BM25 + embeddings + RRF, from
     08_hybrid_search.py) to narrow potentially thousands of chunks
     down to a small candidate pool (this file uses top 8).
  2. Only THEN run the slow, accurate cross-encoder — but now it only
     has to score 8 pairs, not the whole corpus.
  3. Take the cross-encoder's top few (this file uses top 3) as the
     final context for the LLM.

This is a two-stage funnel: retrieval optimizes for RECALL cheaply
("get the right chunk somewhere in the top 8"), reranking optimizes for
PRECISION expensively but on a small set ("of those 8, put the actual
best ones first").

Pipeline for this script (steps A-D are identical to 08_hybrid_search.py):
  load docs -> split into chunks -> build BM25 index + build FAISS index
  -> given a question, retrieve top-8 from EACH -> fuse with RRF into a
  candidate pool -> [NEW] rerank that pool with a cross-encoder -> take
  the reranked top-3 -> stuff into a prompt -> ask the LLM.
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


# --- Step A: load documents (identical to 08_hybrid_search.py) ---
documents = []
for file_path in sorted(DOCS_DIR.glob("*.txt")):
    documents.extend(TextLoader(str(file_path)).load())

print(f"Loaded {len(documents)} document(s) from {DOCS_DIR}")


# --- Step B: split into chunks (identical to 08_hybrid_search.py) ---
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

print(f"Split into {len(chunks)} chunk(s)\n")


# --- Step C: build the two candidate-generation retrievers ---
# CANDIDATE_POOL_SIZE is deliberately wider than 08_hybrid_search.py's
# k=5 — reranking works best with more raw material to sort through.
# Since the cross-encoder step below is the expensive one, we keep this
# pool small enough to rerank quickly (8 candidates, not 50).
CANDIDATE_POOL_SIZE = 8
FINAL_TOP_N = 3

bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = CANDIDATE_POOL_SIZE

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": CANDIDATE_POOL_SIZE})


# --- Step D: Reciprocal Rank Fusion (identical logic to 08_hybrid_search.py) ---
def reciprocal_rank_fusion(ranked_lists: list[list], k: int = 60, top_n: int = CANDIDATE_POOL_SIZE) -> list:
    """Merge multiple ranked lists of Documents into one, via RRF. Here it
    produces the CANDIDATE POOL for reranking, not the final answer set —
    top_n is CANDIDATE_POOL_SIZE (8), not FINAL_TOP_N (3)."""
    scores: dict[str, float] = {}
    doc_by_key: dict[str, object] = {}

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list):
            key = doc.page_content
            doc_by_key[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)

    fused_keys = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [doc_by_key[key] for key in fused_keys[:top_n]]


# --- Step E: the reranker ---
# "cross-encoder/ms-marco-MiniLM-L-6-v2" is a small (~80MB), free,
# well-regarded cross-encoder trained specifically for search relevance
# ranking (MS MARCO is a large real-world search-query dataset). Same
# "small local model, downloads once, runs offline after" story as the
# HuggingFaceEmbeddings model above — no new API key needed.
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(question: str, candidates: list, top_n: int = FINAL_TOP_N) -> list:
    """Score each (question, candidate_text) PAIR together with the
    cross-encoder, then return the top_n candidates sorted by that score.

    Unlike RRF (which combines RANKS from two retrievers that never saw
    the query and document together), this produces a genuine relevance
    score computed by a model that read the query and each document
    JOINTLY — that's the whole point of a cross-encoder.
    """
    # predict() wants a list of (text_a, text_b) tuples — one tuple per
    # pair to score. Order matters for some cross-encoders (query first,
    # document second, matching how the model was trained).
    pairs = [(question, doc.page_content) for doc in candidates]
    scores = reranker.predict(pairs)

    # Sort candidates by their cross-encoder score, descending — this is
    # the actual RERANKING step: the input order (from RRF) is discarded
    # and replaced by this model's own judgment of relevance.
    scored = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
    return scored[:top_n]


# --- Step F: build the RAG chain (same prompt pattern as 08_hybrid_search.py) ---
prompt = ChatPromptTemplate.from_template(
    """Answer the question using ONLY the context below. If the answer
isn't in the context, say "I don't have that information" — do not guess.

Context:
{context}

Question: {question}

Answer:"""
)

llm = ChatGroq(model="openai/gpt-oss-20b")


def format_docs(docs):
    return "\n\n".join(f"[{Path(d.metadata['source']).name}]\n{d.page_content}" for d in docs)


# --- Step G: ask questions, using HYBRID retrieval + RERANKING ---
def ask(question: str):
    bm25_results = bm25_retriever.invoke(question)
    semantic_results = semantic_retriever.invoke(question)

    candidate_pool = reciprocal_rank_fusion([bm25_results, semantic_results])
    reranked = rerank(question, candidate_pool)

    print(f"Q: {question}")
    print(
        "RRF candidate pool:   ",
        [Path(d.metadata["source"]).name for d in candidate_pool],
    )
    print(
        "Reranked (final) top:  ",
        [(Path(d.metadata["source"]).name, f"{score:.2f}") for d, score in reranked],
    )

    final_chunks = [doc for doc, _score in reranked]

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "context": format_docs(final_chunks),
        "question": question,
    })

    print(f"A: {answer}\n")
    print("-" * 60)


if __name__ == "__main__":
    # A question phrased so it doesn't obviously favor one chunk over
    # another at the RRF stage is the most informative case to watch —
    # compare "RRF candidate pool" order (rank-based, blind to actual
    # query/document interaction) against "Reranked (final) top" order
    # (cross-encoder scores, printed alongside each pick) to see the
    # reranker actually change or confirm the ordering.
    ask("Who is the author of the company leave policy?")
    ask("What foods should I eat to stay healthy?")
