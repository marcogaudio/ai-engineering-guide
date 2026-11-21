# AI Engineering Guide

[![CI](https://github.com/marcogaudio/ai-engineering-guide/actions/workflows/ci.yml/badge.svg)](https://github.com/marcogaudio/ai-engineering-guide/actions/workflows/ci.yml)

A practical guide to AI engineering in Python. Each chapter explains the concepts briefly, with code snippets, and comes with **runnable examples that are tested on every push**.

The goal: go from "I can call an LLM API" to building reliable, production-grade LLM applications such as RAG systems, agents and evals.

## Chapters

| # | Chapter | Topics | Status |
|---|---|---|---|
| 01 | [Python foundations](chapters/01-python-foundations/README.md) | type hints · dataclasses vs Pydantic · venv & uv · async/await · generators & streaming | ✅ |
| 02 | [asyncio in depth](chapters/02-asyncio/README.md) | coroutines · event loop · gather / as_completed · queues · async generators · TaskGroup · ExceptionGroup | ✅ |
| 03 | Calling LLM APIs | messages & system prompts · parameters · tokens & cost · retries & rate limits | 🔜 |
| 04 | Structured outputs & tool use | JSON schema · Pydantic validation · tool calling loops | 🔜 |
| 05 | Embeddings & vector search | embeddings · similarity · vector stores · chunking | 🔜 |
| 06 | RAG | retrieval pipelines · reranking · citations · failure modes | 🔜 |
| 07 | Evals | test sets · graders (exact, LLM-as-judge) · regression tracking | 🔜 |
| 08 | Agents | agent loop · tools · memory · guardrails | 🔜 |
| 09 | Observability & cost | logging · tracing · token and latency budgets | 🔜 |
| 10 | Deployment | FastAPI · streaming endpoints · Docker · secrets | 🔜 |

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

## Repository layout

```
chapters/
  01-python-foundations/
    README.md          ← the chapter: concepts + snippets
    examples/*.py      ← runnable, tested examples
  02-asyncio/
    ...
  TEMPLATE.md          ← copy this to start a new chapter
tests/
  test_examples.py     ← runs every example script; fails if one crashes
.github/workflows/ci.yml
```

## Adding a chapter

1. Copy `chapters/TEMPLATE.md` to `chapters/NN-topic/README.md`.
2. Put runnable scripts in `chapters/NN-topic/examples/`. Each must run with `uv run <file>` and exit with code 0.
3. Add the chapter to the table above. `uv run pytest` picks up the new examples automatically.
