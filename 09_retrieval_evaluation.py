"""
STEP 9: Retrieval evaluation — a golden test set + Precision@k, Recall@k,
MRR, and Hit Rate@k, run against BM25-only, semantic-only, RRF-fused
(from 08_hybrid_search.py), and RRF+reranked (from 10_reranking.py), so
"is hybrid actually better, and does reranking help on top of THAT" gets
a real number instead of eyeballing printed lists.

WHY THIS FILE EXISTS:
08_hybrid_search.py prints "BM25 top matches" vs "Semantic top matches"
vs "Fused matches" and lets you eyeball whether they differ. That's fine
for a demo, but it doesn't answer the actual question: for the KINDS of
questions your users will ask, does hybrid search retrieve the RIGHT
chunk more often than either method alone? You can't answer that without
(1) knowing what the "right" chunk actually is for a set of questions,
and (2) a repeatable score to compare retrievers by. That's what
evaluation is — everything below exists to turn "looks better" into
"measurably better, by X%."

THE GOLDEN TEST SET:
A list of (question, expected_source_file) pairs, hand-written by
reading docs/*.txt and asking "what would a real user ask that this
file answers?" This is the ground truth every metric below is computed
against — without it, there's nothing to score retrieval accuracy
AGAINST. Hand-labeling ~10-20 questions like this is standard for a
small project; production systems either scale this up with synthetic
LLM-generated questions per chunk, or mine real user queries + which
result they found useful.

Two of the questions below are deliberately adversarial, not just
easy lookups:
  - Q8 hinges on an exact number (₹499) that BM25's keyword match is
    well suited to, while a paraphrase-heavy embedding search might
    blur it against the other pricing tiers.
  - Q9 is phrased with NONE of the source document's exact words
    ("time off rules" vs the doc's "leave policy"), which pure BM25
    is likely to miss entirely since it can't reason about meaning.
These two are what should make BM25-only and semantic-only diverge in
the results table -- and are exactly the cases hybrid/RRF exists for.

THE METRICS (all computed by hand, no library — see each function's
docstring for the definition):
  - Precision@k — of the k chunks returned, how many were relevant?
  - Recall@k    — of the relevant chunks that exist, how many did we find?
  - MRR         — how high up was the FIRST relevant chunk?
  - Hit Rate@k  — binary: was there at least one relevant chunk in top-k?

This project only labels ONE relevant chunk per question (the source
file the question was written from), so Precision@k and Recall@k will
usually move together here — with a richer multi-relevant-chunk-per-
question golden set they can diverge more informatively. All four are
still shown because each is standard in production retrieval evals and
answers a subtly different question.
"""

from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder

load_dotenv()

DOCS_DIR = Path(__file__).parent / "docs"


# --- Step A: load + split docs (identical to 08_hybrid_search.py) ---
documents = []
for file_path in sorted(DOCS_DIR.glob("*.txt")):
    documents.extend(TextLoader(str(file_path)).load())

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

print(f"Loaded {len(documents)} document(s), split into {len(chunks)} chunk(s)\n")


# --- Step B: build the retrievers under test ---
# k=8 (not 5) so the reranked evaluation below has a genuinely wide
# candidate pool per retriever to choose from before fusing + reranking
# — matching 10_reranking.py's CANDIDATE_POOL_SIZE. The BM25-only and
# Semantic-only rows still score against just their top-K (see `evaluate`
# below, which slices retrieved_sources[:k] itself), so returning more
# candidates here doesn't change those two rows' results.
RETRIEVER_K = 8

bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = RETRIEVER_K

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})


def reciprocal_rank_fusion(ranked_lists: list[list], k: int = 60, top_n: int = 5) -> list:
    """Same RRF logic as 08_hybrid_search.py — merges ranked lists by rank,
    not raw score, so BM25 and embedding scores never need to be compared
    on the same scale."""
    scores: dict[str, float] = {}
    doc_by_key: dict[str, object] = {}
    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list):
            key = doc.page_content
            doc_by_key[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    fused_keys = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [doc_by_key[key] for key in fused_keys[:top_n]]


# --- Step B2: the reranker under test (same model as 10_reranking.py) ---
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(question: str, candidates: list, top_n: int = 5) -> list:
    """Same reranking logic as 10_reranking.py — score each (question,
    candidate) PAIR jointly with the cross-encoder, then sort by that
    score. Discards RRF's rank-based ordering entirely in favor of this
    model's own relevance judgment."""
    if not candidates:
        return []
    pairs = [(question, doc.page_content) for doc in candidates]
    scores = reranker.predict(pairs)
    scored = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
    return [doc for doc, _score in scored[:top_n]]


# --- Step C: the golden test set ---
# Each entry: (question, expected_source_filename). The filename is the
# ground truth — a retrieved chunk counts as "relevant" if it came from
# this file. Read straight from docs/*.txt content above.
GOLDEN_SET = [
    ("How many days of paid annual leave do full-time employees get?", "company_leave_policy.txt"),
    ("What percentage of daily calories should come from carbohydrates?", "healthy_diet.txt"),
    ("Who invented the World Wide Web and in what year?", "history_internet.txt"),
    ("How much does the Pro plan cost per month?", "product_pricing.txt"),
    ("What keyword does Python use to define a function?", "python_basics.txt"),
    ("Which planet has the most prominent ring system?", "solar_system.txt"),
    ("What protocol did ARPANET adopt in 1983?", "history_internet.txt"),
    ("What is the exact monthly price of the Starter plan?", "product_pricing.txt"),
    ("What are the time off rules for employees at this company?", "company_leave_policy.txt"),
    ("Which planets are called gas giants?", "solar_system.txt"),
]


# --- Step D: the metrics, computed by hand ---
def precision_at_k(retrieved_sources: list[str], relevant_source: str, k: int) -> float:
    """Of the top-k retrieved chunks, what fraction came from the relevant
    source file? Answers: "how much noise came with the right answer?" """
    top_k = retrieved_sources[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for s in top_k if s == relevant_source)
    return hits / len(top_k)


def recall_at_k(retrieved_sources: list[str], relevant_source: str, k: int) -> float:
    """Was the relevant source found anywhere in the top-k? With exactly
    one relevant chunk-source per question (this golden set's design),
    recall@k collapses to 0 or 1 — "did we find it at all?" With multiple
    relevant chunks per question it would be a fraction, like precision."""
    top_k = retrieved_sources[:k]
    return 1.0 if relevant_source in top_k else 0.0


def reciprocal_rank(retrieved_sources: list[str], relevant_source: str) -> float:
    """1 / (rank of the FIRST relevant hit, 1-indexed). Rewards ranking the
    right answer EARLY, not just including it somewhere in the list.
    0 if the relevant source never appears at all."""
    for rank, source in enumerate(retrieved_sources, start=1):
        if source == relevant_source:
            return 1.0 / rank
    return 0.0


def hit_rate_at_k(retrieved_sources: list[str], relevant_source: str, k: int) -> float:
    """Binary version of recall@k — 1 if the relevant source is anywhere
    in top-k, else 0. With one relevant source per question this is
    IDENTICAL to recall_at_k here; the distinction matters once a question
    can have several correct chunks (recall becomes fractional, hit rate
    stays binary "did we get at least one")."""
    return recall_at_k(retrieved_sources, relevant_source, k)


# --- Step E: run every retriever over the golden set, average the metrics ---
def evaluate(retriever_name: str, retrieve_fn, k: int = 3) -> dict:
    """retrieve_fn: (question) -> list of LangChain Documents, best-match-first."""
    precisions, recalls, rrs, hit_rates = [], [], [], []

    for question, relevant_source in GOLDEN_SET:
        docs = retrieve_fn(question)
        retrieved_sources = [Path(d.metadata["source"]).name for d in docs]

        precisions.append(precision_at_k(retrieved_sources, relevant_source, k))
        recalls.append(recall_at_k(retrieved_sources, relevant_source, k))
        rrs.append(reciprocal_rank(retrieved_sources, relevant_source))
        hit_rates.append(hit_rate_at_k(retrieved_sources, relevant_source, k))

    n = len(GOLDEN_SET)
    return {
        "retriever": retriever_name,
        f"Precision@{k}": sum(precisions) / n,
        f"Recall@{k}": sum(recalls) / n,
        "MRR": sum(rrs) / n,
        f"HitRate@{k}": sum(hit_rates) / n,
    }


if __name__ == "__main__":
    K = 3

    # RRF's candidate pool for reranking is wider than K (same reasoning
    # as 10_reranking.py's CANDIDATE_POOL_SIZE) — reranking needs several
    # candidates to actually choose among, not just the final K. Reuses
    # RETRIEVER_K so the pool size can't drift out of sync with how many
    # candidates each retriever above actually returns.
    RERANK_POOL_SIZE = RETRIEVER_K

    results = [
        evaluate("BM25 only", bm25_retriever.invoke, k=K),
        evaluate("Semantic only", semantic_retriever.invoke, k=K),
        evaluate(
            "Hybrid (RRF)",
            lambda q: reciprocal_rank_fusion(
                [bm25_retriever.invoke(q), semantic_retriever.invoke(q)], top_n=K
            ),
            k=K,
        ),
        evaluate(
            "Hybrid+Reranked",
            lambda q: rerank(
                q,
                reciprocal_rank_fusion(
                    [bm25_retriever.invoke(q), semantic_retriever.invoke(q)],
                    top_n=RERANK_POOL_SIZE,
                ),
                top_n=K,
            ),
            k=K,
        ),
    ]

    header = f"{'Retriever':<16} {'Precision@'+str(K):<13} {'Recall@'+str(K):<11} {'MRR':<8} {'HitRate@'+str(K):<10}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['retriever']:<16} "
            f"{r[f'Precision@{K}']:<13.2f} "
            f"{r[f'Recall@{K}']:<11.2f} "
            f"{r['MRR']:<8.2f} "
            f"{r[f'HitRate@{K}']:<10.2f}"
        )

    print(f"\nScored against {len(GOLDEN_SET)} golden questions, top-{K} results per retriever.")
