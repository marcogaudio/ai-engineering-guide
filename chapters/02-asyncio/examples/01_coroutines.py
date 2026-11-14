"""Coroutines, async def/await, and the event loop: sync vs async timing.

Run: uv run chapters/02-asyncio/examples/01_coroutines.py
"""

import asyncio
import time


def count_sync(name: str) -> None:
    print(f"{name}: one")
    time.sleep(1)  # blocks the whole program
    print(f"{name}: two")


async def count(name: str) -> None:
    print(f"{name}: one")
    await asyncio.sleep(1)  # pauses this coroutine; the event loop runs the others
    print(f"{name}: two")


async def main() -> None:
    await asyncio.gather(count("A"), count("B"), count("C"))


if __name__ == "__main__":
    start = time.perf_counter()
    for n in "ABC":
        count_sync(n)
    print(f"sync:  {time.perf_counter() - start:.2f}s\n")

    coro = main()
    print("calling main() only creates:", coro)  # nothing has run yet
    start = time.perf_counter()
    asyncio.run(coro)  # creates the event loop, runs it to completion, closes it
    print(f"async: {time.perf_counter() - start:.2f}s")
