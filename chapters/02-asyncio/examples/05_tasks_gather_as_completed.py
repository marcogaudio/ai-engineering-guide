"""create_task vs await, gather (input order) vs as_completed (finish order).

Run: uv run chapters/02-asyncio/examples/05_tasks_gather_as_completed.py
"""

import asyncio
import time


async def work(name: str, secs: float) -> str:
    await asyncio.sleep(secs)
    return f"{name} done in {secs}s"


async def main() -> None:
    # Common mistake: awaiting one by one runs them sequentially
    start = time.perf_counter()
    await work("A", 1)
    await work("B", 0.5)
    print(f"await one by one: {time.perf_counter() - start:.2f}s")

    # create_task: both start immediately in the background
    start = time.perf_counter()
    t1 = asyncio.create_task(work("A", 1))
    t2 = asyncio.create_task(work("B", 0.5))
    print(await t1, "|", await t2, f"| {time.perf_counter() - start:.2f}s")

    # gather: results in INPUT order
    print("gather:", await asyncio.gather(work("A", 1), work("B", 0.5)))

    # as_completed: results in FINISH order
    print("as_completed:", end=" ")
    for fut in asyncio.as_completed([work("A", 1), work("B", 0.5)]):
        print(await fut, end="; ")
    print()


if __name__ == "__main__":
    asyncio.run(main())
