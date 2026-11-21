"""Async calls to a real public REST API: package info from PyPI, fetched concurrently.

The PyPI JSON API needs no key: GET https://pypi.org/pypi/<package>/json returns a JSON object.

requires: network  (skipped in tests unless RUN_NETWORK_EXAMPLES=1)
Run: uv run chapters/03-rest-apis-and-secrets/examples/05_async_public_api.py
"""

import asyncio
import time

import httpx2
from pydantic import BaseModel

PACKAGES = ["httpx2", "pydantic", "pydantic-settings", "python-dotenv", "openai", "anthropic"]


# The response has dozens of fields: declare only the ones we use. Pydantic ignores the rest.
class PackageInfo(BaseModel):
    name: str
    version: str  # latest release
    summary: str | None = None
    requires_python: str | None = None


async def fetch_info(client: httpx2.AsyncClient, package: str) -> PackageInfo:
    # `await`: while this request waits on the network, the event loop sends the others.
    response = await client.get(f"/pypi/{package}/json")
    response.raise_for_status()  # a typo in a package name -> 404 -> HTTPStatusError
    # The interesting part is nested under the "info" key of the JSON object.
    return PackageInfo.model_validate(response.json()["info"])


async def main() -> None:
    # AsyncClient is the async twin of Client: same arguments, but its methods are coroutines.
    # One client for all requests, so they share a pool of open connections to pypi.org.
    async with httpx2.AsyncClient(
        base_url="https://pypi.org",
        timeout=15,
        # Identify your program: polite, and some APIs reject requests without a User-Agent.
        headers={"User-Agent": "ai-engineering-guide-example", "Accept": "application/json"},
    ) as client:
        # Look at one raw response first: status, and the headers a real API sends back.
        first = await client.get(f"/pypi/{PACKAGES[0]}/json")
        print("status:         ", first.http_version, first.status_code, first.reason_phrase)
        for header in ["content-type", "cache-control", "etag", "content-length"]:
            print(f"{header + ':':16}", first.headers.get(header))

        # Then all packages at once with gather (chapter 02): results in input order.
        start = time.perf_counter()
        infos = await asyncio.gather(*(fetch_info(client, p) for p in PACKAGES))
        elapsed = time.perf_counter() - start

    print(f"\n{len(infos)} packages in {elapsed:.2f}s (concurrently)")
    for info in infos:
        print(f"  {info.name:18} {info.version:10} python {info.requires_python or '?':8}"
              f" {(info.summary or '')[:50]}")


if __name__ == "__main__":
    asyncio.run(main())
