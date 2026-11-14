"""async/await for LLM calls: run an eval set concurrently instead of one by one.

Uses a fake LLM (1 s of simulated network latency) so it runs without an API key.
See llm_client_demo.py for the same pattern with the real Anthropic client.

Run: uv run chapters/01-python-foundations/examples/async_llm_calls.py
"""

import asyncio
import random
import time

LATENCY_S = 1.0


async def fake_llm(prompt: str) -> str:
    await asyncio.sleep(LATENCY_S)  # waiting on the network: the event loop runs others
    if random.random() < 0.05:
        raise TimeoutError(f"timed out on {prompt!r}")
    return prompt.upper()


async def run_sequential(prompts: list[str]) -> list[str]:
    return [await fake_llm(p) for p in prompts]  # each await blocks the next call


async def run_concurrent(
    prompts: list[str], max_in_flight: int = 10
) -> list[str | BaseException]:
    sem = asyncio.Semaphore(max_in_flight)  # cap concurrency to respect rate limits

    async def one(prompt: str) -> str:
        async with sem:
            return await fake_llm(prompt)

    # return_exceptions=True: one failure doesn't crash the whole run,
    # and results come back in the same order as the inputs
    return await asyncio.gather(*(one(p) for p in prompts), return_exceptions=True)


async def main() -> None:
    prompts = [f"example {i}" for i in range(50)]

    start = time.perf_counter()
    await run_sequential(prompts[:3])
    print(f"sequential, 3 calls:      {time.perf_counter() - start:.2f}s  (50 calls would be ~50s)")

    for limit in (50, 10):
        start = time.perf_counter()
        results = await run_concurrent(prompts, max_in_flight=limit)
        elapsed = time.perf_counter() - start
        errors = [r for r in results if isinstance(r, BaseException)]
        print(f"concurrent, limit={limit:>2}:   {elapsed:.2f}s  ({len(errors)} errors)")

    for prompt, result in list(zip(prompts, results, strict=True))[:3]:
        print(f"  {prompt!r} -> {result!r}")


if __name__ == "__main__":
    random.seed(0)
    asyncio.run(main())  # in Jupyter, use `await main()` instead
