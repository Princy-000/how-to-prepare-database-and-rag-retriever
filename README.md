# Agentic RAG Retrieval Agent

A retrieval-augmented generation (RAG) agent that searches a knowledge base of Hugging Face documentation and answers questions using retrieved context.

## What It Does

This agent demonstrates **agentic RAG** — instead of a fixed retrieval pipeline, the agent:
- Decides how many documents to retrieve
- Filters by documentation source (e.g., `transformers`, `peft`, `datasets`)
- Reformulates queries if results are poor
- Falls back to its own knowledge if retrieval fails

## How It Works

1. **Dataset** — Loads `m-ric/huggingface_doc` (2,647 documentation pages)
2. **Splitting** — Breaks documents into chunks using `RecursiveCharacterTextSplitter`
3. **Embeddings** — Converts chunks to vectors using `thenlper/gte-small`
4. **Vector Database** — Stores embeddings in FAISS for similarity search
5. **Retriever Tool** — A custom `Tool` class that wraps the FAISS database
6. **Agent** — A `ToolCallingAgent` that uses the retriever to answer questions

## Tech Stack

| Component | Technology |
|:---|:---|
| Agent framework | `smolagents` |
| LLM | DeepSeek-V4.1-Flash |
| Embeddings | `thenlper/gte-small` |
| Vector database | FAISS |
| Document processing | LangChain |
| Dataset | `m-ric/huggingface_doc` |

## Example

**Query:** "Please show me a LORA finetuning script"

**What the agent does:**
1. Calls the retriever with a specific source filter
2. If results are insufficient, tries again with fewer restrictions
3. Uses retrieved docs to answer — or falls back to its own knowledge if the docs don't contain the answer

## Setup

```bash
pip install smolagents datasets langchain langchain-community langchain-text-splitters sentence-transformers faiss-cpu
