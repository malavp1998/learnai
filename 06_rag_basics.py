"""
STEP 6: RAG (Retrieval-Augmented Generation) — the basics.

Problem: an LLM only knows what it was trained on. It has never seen
your company's leave policy, your product's pricing, or your personal
notes. RAG fixes this WITHOUT retraining the model — instead, at
question time, we:

  1. Find the chunks of YOUR documents most relevant to the question
     (using semantic search over "embeddings", not keyword matching).
  2. Paste those chunks into the prompt as context.
  3. Ask the LLM to answer USING that context.

This is why RAG apps can answer questions about private, recent, or
niche data the model was never trained on.

Pipeline for this script:
  load docs -> split into chunks -> embed chunks -> store in a vector
  database (FAISS) -> given a question, retrieve the top-k most
  similar chunks -> stuff them into a prompt -> ask the LLM.

Note on embeddings: Groq does NOT serve embedding models (it's LLM
inference only). So we use a small, free, LOCAL embedding model from
HuggingFace (runs on your machine, no extra API key, downloads once
then works offline). Groq is still used for the final answer generation.
"""

from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

DOCS_DIR = Path(__file__).parent / "docs"


# --- Step A: load documents ---
# TextLoader reads one file into a LangChain "Document" object, which
# has .page_content (the text) and .metadata (e.g. source filename).
# We load every .txt file in docs/ and combine them into one list.
documents = []
for file_path in sorted(DOCS_DIR.glob("*.txt")):
    documents.extend(TextLoader(str(file_path)).load())

print(f"Loaded {len(documents)} document(s) from {DOCS_DIR}")


# --- Step B: split documents into chunks ---
# Why split at all? Two reasons:
#   1. LLM prompts have a size limit — you can't paste 20 full documents in.
#   2. Smaller chunks give more PRECISE retrieval — if a whole 500-line
#      doc is one chunk, matching it tells you little about which PART
#      is relevant.
# chunk_size=500 characters, chunk_overlap=50 (chunks share 50 chars at
# the boundary so we don't cut a sentence in half and lose meaning).
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

print(f"Split into {len(chunks)} chunk(s)\n")


# --- Step C: embed chunks + store them in a vector database ---
# An embedding turns text into a list of numbers (a vector) that
# captures its MEANING. Texts with similar meaning end up as vectors
# that are close together in that number-space. This is what makes
# semantic search possible — it matches on meaning, not exact words.
#
# "all-MiniLM-L6-v2" is a small (~80MB), fast, well-regarded free
# embedding model. First run downloads it; later runs use the local
# cache.
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# FAISS is an in-memory vector database (from Meta/Facebook AI). It
# stores every chunk's embedding and lets us efficiently find the
# closest ones to a query embedding. Rebuilt fresh every run here —
# in a real app you'd save/load this to disk instead of recomputing.
vectorstore = FAISS.from_documents(chunks, embedding=embeddings)

# A "retriever" wraps the vector store with a simple interface:
# give it a query, get back the top-k most similar chunks.
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# --- Step D: build the RAG chain ---
# The prompt has a {context} slot for retrieved chunks and a
# {question} slot for the user's question. We explicitly instruct
# the model to answer ONLY from the context — this reduces (does not
# eliminate) hallucination, since the model is nudged to stick to what
# it was given instead of inventing an answer from its own training.
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
    # Join retrieved chunks into one block of text, each labeled with
    # its source file so you can see (and later verify) where an
    # answer's information actually came from.
    return "\n\n".join(f"[{Path(d.metadata['source']).name}]\n{d.page_content}" for d in docs)


# --- Step E: ask questions ---
def ask(question: str):
    retrieved_chunks = retriever.invoke(question)

    print(f"Q: {question}")
    print("Retrieved chunks from:", [Path(d.metadata["source"]).name for d in retrieved_chunks])

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "context": format_docs(retrieved_chunks),
        "question": question,
    })

    print(f"A: {answer}\n")
    print("-" * 60)


if __name__ == "__main__":
    ask("Who is the author of the company leave policy?")
 