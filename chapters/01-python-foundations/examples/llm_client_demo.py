"""The same patterns with a real LLM, for free.

Uses the `openai` SDK against any OpenAI-compatible API. Defaults to a local Ollama model:
    ollama pull llama3.2 && ollama serve
Run:  uv run chapters/01-python-foundations/examples/llm_client_demo.py

Other free providers, by env vars alone:
    Gemini:      LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
                 LLM_API_KEY=<AI Studio key>  LLM_MODEL=gemini-2.5-flash
    Groq:        LLM_BASE_URL=https://api.groq.com/openai/v1  LLM_API_KEY=...  LLM_MODEL=...
    OpenRouter:  LLM_BASE_URL=https://openrouter.ai/api/v1  LLM_API_KEY=...  LLM_MODEL=<x>:free
Without a reachable server it prints a message and exits, so the tests still pass.
"""

import asyncio
import os
from typing import Literal

# The `openai` SDK isn't tied to OpenAI: it talks to any server that speaks the same HTTP API
# (Ollama, Gemini, Groq, OpenRouter, vLLM...). `OpenAI` is the sync client, `AsyncOpenAI` the
# async one, and `APIConnectionError` is what it raises when the server can't be reached.
from openai import APIConnectionError, AsyncOpenAI, OpenAI
from pydantic import BaseModel, Field

# Configuration comes from the environment, never from hard-coded secrets. The defaults point at
# a local Ollama server, so the demo works with zero setup beyond `ollama serve`.
BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
API_KEY = os.environ.get("LLM_API_KEY", "ollama")  # Ollama ignores it, the SDK requires one
MODEL = os.environ.get("LLM_MODEL", "llama3.2")


# The shape we want the model's answer to have. Pydantic turns this class into a JSON schema
# that is sent to the model, then validates the reply against it:
# - `Literal[...]` restricts `label` to exactly these three strings.
# - `Field(ge=0, le=1)` rejects any score outside the range 0..1.
class Sentiment(BaseModel):
    label: Literal["positive", "negative", "neutral"]
    score: float = Field(ge=0, le=1)


def structured_output(client: OpenAI) -> Sentiment | None:
    """Pydantic model in, validated object out."""
    # `.parse()` is `.create()` plus structured output: passing a Pydantic class as
    # `response_format` makes the server constrain the reply to its JSON schema, and the SDK
    # parses the JSON back into a `Sentiment` instance for us.
    completion = client.chat.completions.parse(
        model=MODEL,
        max_tokens=256,  # upper bound on the reply length (in tokens), a cost/latency guard
        # A conversation is a list of messages, each with a role ("system", "user",
        # "assistant") and its text content. Here it's a single user turn.
        messages=[{"role": "user", "content": "Classify: 'The support team was fantastic'"}],
        response_format=Sentiment,
    )
    # An API call can return several alternative answers (`choices`); we asked for one.
    # `.parsed` is a typed `Sentiment`, or None if the model refused or the reply was cut off.
    return completion.choices[0].message.parsed


def stream_answer(client: OpenAI) -> None:
    """Print tokens as they arrive."""
    # With `stream=True` the call returns immediately with an iterator instead of waiting for
    # the full answer: the server sends small chunks as the model generates them. This is how
    # chat UIs show text appearing word by word.
    stream = client.chat.completions.create(
        model=MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": "Explain retrieval-augmented generation briefly."}],
        stream=True,
        # Ask the server to append token usage to the stream (off by default when streaming).
        stream_options={"include_usage": True},
    )
    output_tokens = None
    for chunk in stream:
        # Each chunk carries a `delta`: just the new piece of text, not the whole answer so far.
        # Some chunks have no text (e.g. the first one, or the final usage-only chunk).
        if chunk.choices and chunk.choices[0].delta.content:
            # `end=""` avoids a newline after every piece; `flush=True` prints it right away
            # instead of waiting for Python's output buffer to fill.
            print(chunk.choices[0].delta.content, end="", flush=True)
        if chunk.usage:  # sent in the last chunk
            output_tokens = chunk.usage.completion_tokens
    print(f"\n[output tokens: {output_tokens}]")


async def concurrent_eval(questions: list[str]) -> list[str | BaseException]:
    """Run many calls concurrently, at most 5 at a time."""
    # The async client: its methods are coroutines, so while one request waits on the network
    # the event loop can send the others. One client is shared by all requests.
    client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
    # A semaphore is a counter of free slots: at most 5 requests are in flight at once, the
    # rest wait their turn. This keeps us under provider rate limits.
    sem = asyncio.Semaphore(5)

    async def ask(question: str) -> str:
        async with sem:  # take a slot, release it automatically when the block ends
            # `await` pauses this coroutine (not the whole program) until the reply arrives.
            completion = await client.chat.completions.create(
                model=MODEL,
                max_tokens=100,
                messages=[{"role": "user", "content": question}],
            )
            # `content` can be None (e.g. a tool-call reply), so fall back to an empty string.
            return completion.choices[0].message.content or ""

    # Start one `ask()` per question and wait for all of them. Results come back in the same
    # order as `questions`. With `return_exceptions=True` a failed call becomes an exception
    # object in the list instead of cancelling the whole batch.
    return await asyncio.gather(*(ask(q) for q in questions), return_exceptions=True)


if __name__ == "__main__":
    # Health check before the demo: a cheap request that lists the server's models. No retries
    # and a short timeout, so a missing server fails fast instead of hanging.
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY, max_retries=0, timeout=5)
    try:
        client.models.list()
    except APIConnectionError:
        # Exit with code 0 (success): no server is an expected situation, not a bug, and the
        # test suite runs every example. `from None` hides the original traceback.
        print(f"No LLM server reachable at {BASE_URL}: skipping the live demo.")
        raise SystemExit(0) from None

    # The real client keeps the SDK defaults (automatic retries, long timeout), which suit
    # generation calls that can take a while on a local model.
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    print("structured:", structured_output(client))
    stream_answer(client)
    # `asyncio.run()` starts an event loop, runs the coroutine to completion, then closes it.
    # It's the bridge from normal (sync) code into async code.
    answers = asyncio.run(concurrent_eval(["Capital of Italy?", "2 + 2?", "Largest planet?"]))
    print("concurrent:", answers)
