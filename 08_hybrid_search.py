"""
STEP 8: Hybrid search — BM25 (keyword) + semantic (embedding) search,
combined with Reciprocal Rank Fusion (RRF).

PROBLEM WITH PURE SEMANTIC SEARCH (06_rag_basics.py):
Embeddings match on MEANING, not exact words. That's usually great, but
it can miss queries that hinge on a specific keyword, ID, code, or exact
phrase — e.g. searching for "error code E4021" or a person's exact name.
The embedding model may consider a chunk about "authentication failures"
semantically close even if it never contains "E4021" verbatim, while a
chunk that DOES contain "E4021" but is phrased unusually might rank lower.

PROBLEM WITH PURE KEYWORD SEARCH (BM25 alone):
BM25 (used by classic search engines) matches on literal word overlap
+ term frequency/rarity. It's great at exact terms but blind to meaning
— it won't connect "leave policy" with "time off rules" if the words
don't overlap.

THE FIX — HYBRID SEARCH:
Run BOTH searches over the same chunks, then MERGE the two ranked lists
into one. This gets you exact-term precision AND semantic recall at
once. The merging step is the interesting part, because BM25 scores and
cosine-similarity scores are NOT on the same scale (BM25 might return
7.4, embeddings might return 0.83) — you can't just average them
directly. Reciprocal Rank Fusion (RRF) sidesteps this entirely.

RECIPROCAL RANK FUSION (RRF):
Instead of combining raw SCORES, RRF combines RANKS (1st place, 2nd
place, ...) — which are already on the same scale no matter what
scoring system produced them. For each document, in each ranked list it
appears in:

    rrf_score += 1 / (k + rank)

k is a constant (60 is the standard default from the original RRF
paper) that dampens the impact of very top-ranked results so one list's
#1 pick doesn't automatically dominate. A document that ranks well in
BOTH lists accumulates a higher combined score than one that ranks
#1 in only one list and is absent from the other. Sum across all
lists the document appears in, then sort by that combined score,
descending.

Pipeline for this script (steps A-C are identical to 06_rag_basics.py):
  load docs -> split into chunks -> [build BM25 index] AND
  [embed chunks -> build FAISS index] -> given a question, retrieve
  top-k from EACH -> fuse rankings with RRF -> stuff fused top-k into
  a prompt -> ask the LLM.
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

load_dotenv()

DOCS_DIR = Path(__file__).parent / "docs"


# --- Step A: load documents (same as 06_rag_basics.py) ---
documents = []
for file_path in sorted(DOCS_DIR.glob("*.txt")):
    documents.extend(TextLoader(str(file_path)).load())

print(f"Loaded {len(documents)} document(s) from {DOCS_DIR}")


# --- Step B: split into chunks (same settings as 06_rag_basics.py) ---
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

print(f"Split into {len(chunks)} chunk(s)\n")


# --- Step C1: build the KEYWORD retriever (BM25) ---
# BM25Retriever indexes chunks by their literal words — no embedding
# model involved, no API call, pure statistics over term frequency.
# It ranks chunks by how well their words match the query's words,
# weighting RARE words (across the whole corpus) higher than common
# ones — "the" barely matters, "E4021" matters a lot.
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 5  # how many chunks BM25 returns per query


# --- Step C2: build the SEMANTIC retriever (FAISS + embeddings) ---
# Identical to 06_rag_basics.py — turns each chunk into a vector that
# captures meaning, stored in an in-memory FAISS index.
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})


# --- Step D: Reciprocal Rank Fusion ---
# Takes several ranked lists of the SAME kind of item (here: LangChain
# Document chunks) and merges them into one fused ranking. Works with
# any number of input lists, not just two.
def reciprocal_rank_fusion(ranked_lists: list[list], k: int = 60, top_n: int = 3) -> list:
    """Merge multiple ranked lists of Documents into one, via RRF.

    Args:
        ranked_lists: one list of Documents per retriever, each already
            sorted best-match-first (rank 0 = best).
        k: RRF damping constant. 60 is the standard default — higher
            k flattens the score gap between top and lower ranks.
        top_n: how many fused results to return.
    """
    # We key documents by their text content, since the SAME chunk can
    # (and often does) appear in both the BM25 and semantic results —
    # that overlap is exactly the signal RRF rewards.
    scores: dict[str, float] = {}
    doc_by_key: dict[str, object] = {}

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list):
            key = doc.page_content
            doc_by_key[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)

    fused_keys = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [doc_by_key[key] for key in fused_keys[:top_n]], scores


# --- Step E: build the RAG chain (same prompt pattern as 06_rag_basics.py) ---
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


# --- Step F: ask questions, using HYBRID retrieval ---
def ask(question: str):
    bm25_results = bm25_retriever.invoke(question)
    semantic_results = semantic_retriever.invoke(question)

    fused_chunks, scores = reciprocal_rank_fusion(
        [bm25_results, semantic_results], top_n=3
    )

    print(f"Q: {question}")
    print("BM25 top matches:     ", [Path(d.metadata["source"]).name for d in bm25_results])
    print("Semantic top matches: ", [Path(d.metadata["source"]).name for d in semantic_results])
    print("Fused (RRF) matches:  ", [Path(d.metadata["source"]).name for d in fused_chunks])

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "context": format_docs(fused_chunks),
        "question": question,
    })

    print(f"A: {answer}\n")
    print("-" * 60)


if __name__ == "__main__":
    # A question that leans KEYWORD-heavy (an exact term/name) tends to
    # benefit more from BM25's contribution to the fused ranking, while
    # a question phrased conceptually leans on the semantic retriever.
    # Try both kinds and compare "BM25 top matches" vs "Semantic top
    # matches" vs "Fused" in the printed output to see them diverge.
    ask("Who is the author of the company leave policy?")
    ask("What foods should I eat to stay healthy?")
