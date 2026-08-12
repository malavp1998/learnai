# Learn LangChain — Step by Step

A minimal, hands-on path to learning LangChain using Groq (free, fast LLM API).

## Setup (do this once)

```bash
# 1. Activate the virtual environment
source venv/bin/activate

# 2. Get a free API key: https://console.groq.com/keys
cp .env.example .env
# then edit .env and paste your key in place of "your_groq_api_key_here"
```

## Run the examples in order

Each file builds on the last. Read the comments at the top of each one before running.

```bash
python 01_basic_call.py        # simplest possible call to an LLM
python 02_prompt_template.py   # reusable prompt templates + chaining with |
python 03_chain_with_parser.py # turning raw LLM output into clean Python data
python 04_memory_chat.py       # a real terminal chatbot with conversation memory
python 05_tool_calling.py      # model calls a Python function, ToolMessage carries the result back
python 06_rag_basics.py        # chat with your own documents (RAG)
```

## What each step teaches

| File | Concept |
|---|---|
| `01_basic_call.py` | The model wrapper (`ChatGroq`) and `.invoke()` |
| `02_prompt_template.py` | `ChatPromptTemplate` + LCEL chaining with `\|` |
| `03_chain_with_parser.py` | Output parsers (`StrOutputParser`, list parser) |
| `04_memory_chat.py` | Manually managing conversation history for a chatbot |
| `05_tool_calling.py` | `@tool`, `bind_tools`, `tool_calls`, `ToolMessage` |
| `06_rag_basics.py` | Embeddings, chunking, FAISS vector store, retrieval |

See [NOTES.md](NOTES.md) for revision notes and interview Q&A on all of the above.

## `docs/` folder

Six small sample documents (`company_leave_policy.txt`, `product_pricing.txt`, `python_basics.txt`, `solar_system.txt`, `healthy_diet.txt`, `history_internet.txt`) used as the knowledge base for `06_rag_basics.py`. Deliberately spans unrelated topics so you can see retrieval correctly pick the right document per question.

## Production RAG (in progress)

`06_rag_basics.py` is the "happy path" — real systems have to handle bad retrieval, hallucination, evaluation, and scale. See [RAG_PRODUCTION.md](RAG_PRODUCTION.md) for the full map of production RAG problems, the standard industry fixes, and interview Q&A for each. Scripts `07+` implement these one at a time:

| Script | Topic |
|---|---|
| `07_rag_evaluation.py` | Golden dataset + evaluation metrics (faithfulness, context recall) |
| `08_rag_hybrid_search.py` | BM25 keyword search + vector search fusion |
| `09_rag_reranking.py` | Retrieve-wide-then-rerank pattern |
| `10_rag_citations_grounding.py` | Citation tagging + faithfulness/groundedness check |
| `11_rag_guardrails.py` | Similarity threshold cutoff + prompt injection defense |

## What's next after that

- **Agents (LangGraph)** — let the model loop over multiple tool calls autonomously using a proper framework instead of the hand-written loop in `05_tool_calling.py`
- **Structured output** — force reliable JSON/Pydantic output instead of free text
- **Streaming** — print tokens as they arrive instead of waiting for the full response

Ask when you're ready and we'll add the next script the same way.
