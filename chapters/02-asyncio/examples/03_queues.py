"""Producer/consumer with asyncio.Queue and "poison pill" shutdown.

Run: uv run chapters/02-asyncio/examples/03_queues.py
"""

import asyncio

N_CONSUMERS = 3


async def producer(q: asyncio.Queue[int | None], n_items: int) -> None:
    for i in range(n_items):
        await asyncio.sleep(0.1)  # e.g. pages of documents arriving from an API
        await q.put(i)
        print(f"produced {i}")
    for _ in range(N_CONSUMERS):
        await q.put(None)  # one sentinel ("poison pill") per consumer


async def consumer(name: str, q: asyncio.Queue[int | None]) -> None:
    while (item := await q.get()) is not None:
        await asyncio.sleep(0.3)  # e.g. embed or summarize the item
        print(f"  {name} processed {item}")
    print(f"  {name} stopping")


async def main() -> None:
    q: asyncio.Queue[int | None] = asyncio.Queue()
    await asyncio.gather(
        producer(q, n_items=6),
        *(consumer(f"consumer-{i}", q) for i in range(N_CONSUMERS)),
    )


if __name__ == "__main__":
    asyncio.run(main())
