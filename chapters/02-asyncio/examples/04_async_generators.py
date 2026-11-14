"""Async generators, async for, and async comprehensions.

Run: uv run chapters/02-asyncio/examples/04_async_generators.py
"""

import asyncio
import time
from collections.abc import AsyncIterator


async def powers_of_two(stop: int) -> AsyncIterator[int]:
    for exp in range(stop):
        yield 2**exp
        await asyncio.sleep(0.2)


async def main() -> None:
    async for p in powers_of_two(5):
        print(p, end=" ")
    print()

    big = [p async for p in powers_of_two(5) if p > 2]
    print("async comprehension:", big)

    # Iteration is still SEQUENTIAL: 5 items x 0.2 s each
    start = time.perf_counter()
    _ = [p async for p in powers_of_two(5)]
    print(f"took {time.perf_counter() - start:.2f}s: async for does not run items concurrently")


if __name__ == "__main__":
    asyncio.run(main())
