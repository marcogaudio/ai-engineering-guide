# AI Engineering Guide

[![CI](https://github.com/marcogaudio/ai-engineering-guide/actions/workflows/ci.yml/badge.svg)](https://github.com/marcogaudio/ai-engineering-guide/actions/workflows/ci.yml)

A practical guide to AI engineering in Python. Each chapter explains the concepts briefly, with code snippets, and comes with **runnable examples that are tested on every push**.

The goal: go from "I can call an LLM API" to building reliable, production-grade LLM applications such as RAG systems, agents and evals.

The guide follows Chip Huyen's framing of AI engineering: building applications on top of foundation models that already exist, not training models from scratch. [Chapter 00](chapters/00-what-is-ai-engineering/README.md) explains what that means, so start there.

## Chapters

| # | Chapter | Topics | Status |
|---|---|---|---|
| 00 | [What is AI engineering?](chapters/00-what-is-ai-engineering/README.md) | foundation models · AI vs ML vs full-stack engineering · the three layers of the AI stack | ✅ |
| 01 | [Python foundations](chapters/01-python-foundations/README.md) | type hints · dataclasses vs Pydantic · venv & uv · async/await · generators & streaming | ✅ |
| 02 | [asyncio in depth](chapters/02-asyncio/README.md) | coroutines · event loop · gather / as_completed · queues · async generators · TaskGroup · ExceptionGroup | ✅ |
| 03 | [REST APIs, JSON & secrets](chapters/03-rest-apis-and-secrets/README.md) | HTTP messages · REST methods & status codes · JSON · httpx2 · errors & timeouts · env vars, `.env` & pydantic-settings · SSE streaming | ✅ |
| 04 | [Deep learning & Transformers](chapters/04-deep-learning-and-transformers/README.md) | neural networks · backpropagation · RNNs vs attention · Transformer architecture · BPE & embeddings · next-token training & scaling laws | ✅ |
| 05 | Calling LLM APIs | messages & system prompts · parameters · tokens & cost · retries & rate limits | 🔜 |
| 06 | Structured outputs & tool use | JSON schema · Pydantic validation · tool calling loops | 🔜 |
| 07 | Embeddings & vector search | embeddings · similarity · vector stores · chunking | 🔜 |
| 08 | RAG | retrieval pipelines · reranking · citations · failure modes | 🔜 |
| 09 | Evals | test sets · graders (exact, LLM-as-judge) · regression tracking | 🔜 |
| 10 | Agents | agent loop · tools · memory · guardrails | 🔜 |
| 11 | Observability & cost | logging · tracing · token and latency budgets | 🔜 |
| 12 | Deployment | FastAPI · streaming endpoints · Docker · secrets | 🔜 |

## Quick start

This repo is a [uv](https://docs.astral.sh/uv/) project (see [Chapter 01 §3](chapters/01-python-foundations/README.md#3-virtual-environments-venv-and-uv)).

```bash
git clone https://github.com/marcogaudio/ai-engineering-guide.git
cd ai-engineering-guide
uv sync                                    # creates .venv with the exact locked versions

uv run chapters/02-asyncio/examples/01_coroutines.py      # run any example
uv run pytest                              # run every example as a test
uv run ruff check . && uv run mypy chapters   # lint + type-check
```

Examples that call a real LLM default to a free local model via [Ollama](https://ollama.com) (`ollama pull llama3.2 && ollama serve`). Point them at any OpenAI-compatible API with `LLM_BASE_URL`, `LLM_API_KEY` and `LLM_MODEL`. Without a reachable server they print a message and exit, so the tests still pass. Examples that need the internet run only with `RUN_NETWORK_EXAMPLES=1`.

To keep those settings (and your API key) out of your shell history, put them in a `.env` file, which git ignores (see [Chapter 03 §9](chapters/03-rest-apis-and-secrets/README.md#9-api-keys-and-environment-variables)):

```bash
cp .env.example .env                       # then fill in your values
uv run --env-file .env chapters/01-python-foundations/examples/llm_client_demo.py
```

## Repository layout

```
chapters/
  00-what-is-ai-engineering/
  01-python-foundations/
    README.md          ← the chapter: concepts + snippets
    examples/*.py      ← runnable, tested examples
  02-asyncio/
  03-rest-apis-and-secrets/
  04-deep-learning-and-transformers/
    ...
  TEMPLATE.md          ← copy this to start a new chapter
tests/
  test_examples.py     ← runs every example script; fails if one crashes
.github/workflows/ci.yml
.env.example           ← copy to .env (git-ignored) for your LLM settings and keys
```

## Adding a chapter

1. Copy `chapters/TEMPLATE.md` to `chapters/NN-topic/README.md`.
2. Put runnable scripts in `chapters/NN-topic/examples/`. Each must run with `uv run <file>` and exit with code 0.
3. Add the chapter to the table above. `uv run pytest` picks up the new examples automatically.
