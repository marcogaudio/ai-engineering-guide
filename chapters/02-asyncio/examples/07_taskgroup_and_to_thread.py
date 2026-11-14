"""Modern tools: TaskGroup (3.11+) and asyncio.to_thread (3.9+).

Run: uv run chapters/02-asyncio/examples/07_taskgroup_and_to_thread.py
"""

import asyncio
import time


async def work(name: str, secs: float) -> str:
    await asyncio.sleep(secs)
    return f"{name} done in {secs}s"


async def fails() -> str:
    await asyncio.sleep(0.2)
    raise RuntimeError("upstream API error")


def blocking_call(secs: float) -> str:
    time.sleep(secs)  # e.g. a sync-only library such as requests
    return f"blocking call finished after {secs}s"


async def main() -> None:
    # TaskGroup: a safer gather. Every task has finished when the block exits.
    async with asyncio.TaskGroup() as tg:
        a = tg.create_task(work("A", 0.5))
        b = tg.create_task(work("B", 0.3))
    print("TaskGroup:", a.result(), "|", b.result())

    # If one task fails, the others are cancelled and errors raise as an ExceptionGroup
    try:
        async with asyncio.TaskGroup() as tg:
            slow = tg.create_task(work("slow", 5))
            tg.create_task(fails())
    except* RuntimeError as eg:
        print("TaskGroup failure:", eg.exceptions, "| slow task cancelled:", slow.cancelled())

    # to_thread: run a blocking function without freezing the event loop
    start = time.perf_counter()
    results = await asyncio.gather(
        asyncio.to_thread(blocking_call, 0.5),
        asyncio.to_thread(blocking_call, 0.5),
        work("async", 0.5),
    )
    print(results, f"| {time.perf_counter() - start:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
