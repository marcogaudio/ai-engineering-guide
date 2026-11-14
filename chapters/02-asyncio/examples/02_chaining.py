"""Chaining coroutines: each step depends on the previous one, but chains run concurrently.

Run: uv run chapters/02-asyncio/examples/02_chaining.py
"""

import asyncio
import random
import time


async def fetch_user(uid: int) -> dict:
    delay = random.uniform(0.5, 1.5)
    await asyncio.sleep(delay)  # simulated network call
    print(f"  fetched user {uid} in {delay:.2f}s")
    return {"id": uid, "name": f"user{uid}"}


async def fetch_posts(user: dict) -> list[str]:
    delay = random.uniform(0.5, 1.5)
    await asyncio.sleep(delay)
    print(f"  fetched posts for {user['name']} in {delay:.2f}s")
    return [f"post {i} by {user['name']}" for i in range(2)]


async def user_with_posts(uid: int) -> list[str]:
    user = await fetch_user(uid)  # step 1
    return await fetch_posts(user)  # step 2 needs the result of step 1


async def main() -> None:
    start = time.perf_counter()
    results = await asyncio.gather(*(user_with_posts(i) for i in range(3)))
    print(f"results: {results}")
    print(f"total: {time.perf_counter() - start:.2f}s (sequential would be the sum of all delays)")


if __name__ == "__main__":
    random.seed(1)
    asyncio.run(main())
