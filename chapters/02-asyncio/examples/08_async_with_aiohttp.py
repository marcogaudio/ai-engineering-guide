"""Async context managers (async with) with aiohttp: check several URLs concurrently.

requires: network  (skipped in tests unless RUN_NETWORK_EXAMPLES=1)
Run: uv run chapters/02-asyncio/examples/08_async_with_aiohttp.py
"""

import asyncio
import time

import aiohttp

URLS = ["https://www.python.org", "https://pypi.org", "https://docs.python.org/3/"]


async def check(session: aiohttp.ClientSession, url: str) -> tuple[str, int]:
    # async with closes the response even if an error is raised
    async with session.get(url) as resp:
        return url, resp.status


async def main() -> None:
    start = time.perf_counter()
    timeout = aiohttp.ClientTimeout(total=15)
    async with aiohttp.ClientSession(timeout=timeout) as session:  # one session, reused
        for url, status in await asyncio.gather(*(check(session, u) for u in URLS)):
            print(f"{status}  {url}")
    print(f"checked {len(URLS)} URLs in {time.perf_counter() - start:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
