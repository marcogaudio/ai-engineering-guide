"""Generators and streaming: consume tokens as they arrive.

Run: uv run chapters/01-python-foundations/examples/streaming.py
"""

import asyncio
import json
import time
from collections.abc import AsyncIterator, Iterable, Iterator


# --- 1. A generator yields values one at a time, lazily ---------------------
def fake_stream(text: str, delay: float = 0.05) -> Iterator[str]:
    for word in text.split():
        time.sleep(delay)  # simulates tokens arriving over the network
        yield word + " "


# --- 2. Async generator: the same idea inside async code (e.g. a web server) -
async def fake_astream(text: str, delay: float = 0.05) -> AsyncIterator[str]:
    for word in text.split():
        await asyncio.sleep(delay)
        yield word + " "


# --- 3. Generator pipelines: process big files one record at a time ---------
def read_jsonl(lines: Iterable[str]) -> Iterator[dict]:
    for line in lines:
        if line.strip():
            yield json.loads(line)


def failures(records: Iterable[dict]) -> Iterator[dict]:
    for record in records:
        if not record["passed"]:
            yield record


async def async_demo() -> None:
    async for chunk in fake_astream("streamed from an async generator"):
        print(chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    gen = fake_stream("The answer is forty two")
    print("generator object:", gen)  # nothing has run yet
    print("next():", repr(next(gen)), repr(next(gen)))

    # Typical streaming loop: print each chunk AND keep the full text
    chunks: list[str] = []
    for chunk in fake_stream("Streaming shows the first tokens almost immediately"):
        print(chunk, end="", flush=True)
        chunks.append(chunk)
    print("\nfull text:", "".join(chunks).strip())

    asyncio.run(async_demo())

    # With a real file: read_jsonl(open("eval_results.jsonl"))
    lines = [
        '{"id": 1, "passed": true}',
        '{"id": 2, "passed": false}',
        '{"id": 3, "passed": false}',
    ]
    print("failed case ids:", [r["id"] for r in failures(read_jsonl(lines))])
