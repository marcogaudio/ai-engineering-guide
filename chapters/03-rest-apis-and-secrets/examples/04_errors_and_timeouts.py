"""When API calls fail: status errors vs transport errors, timeouts, and what to retry.

A fake server (httpx2.MockTransport) misbehaves on purpose, so every kind of failure shows up
offline and in a predictable order. The last case is a real network error: a connection to a
port where nothing is listening.
Full retry loops with exponential backoff and rate limits are in chapter 05 (Calling LLM APIs).

Run: uv run chapters/03-rest-apis-and-secrets/examples/04_errors_and_timeouts.py
"""

import socket
import time
from collections import Counter

import httpx2

# 408 Request Timeout, 429 Too Many Requests, and the 5xx server errors are temporary: the same
# request may succeed later. Any other 4xx means the request itself is wrong (bad key, bad JSON,
# missing resource): sending it again gives the same error, so fix the request instead.
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}

CALLS: Counter[str] = Counter()  # how many times the fake server has seen each path


def flaky_api(request: httpx2.Request) -> httpx2.Response:
    """A fake server that fails the first time on purpose, in a different way on each path."""
    path = request.url.path
    CALLS[path] += 1
    first_call = CALLS[path] == 1
    if path == "/overloaded" and first_call:
        return httpx2.Response(503, json={"error": "server overloaded, try again"})
    if path == "/rate-limited" and first_call:
        # Retry-After tells the client how many seconds to wait before trying again.
        return httpx2.Response(429, headers={"Retry-After": "1"}, json={"error": "slow down"})
    if path == "/private":
        return httpx2.Response(401, json={"error": "invalid API key"})  # fails every time
    if path == "/slow" and first_call:
        # A timeout is not a status code: no response arrived at all, so httpx2 raises.
        raise httpx2.ReadTimeout("the server took too long to answer", request=request)
    return httpx2.Response(200, json={"ok": True, "path": path, "attempt": CALLS[path]})


def is_retryable(error: httpx2.HTTPError) -> bool:
    if isinstance(error, httpx2.HTTPStatusError):  # the server answered, with an error status
        return error.response.status_code in RETRYABLE_STATUS
    # TransportError: timeouts, refused or dropped connections... no usable answer arrived.
    # These are usually temporary too.
    return isinstance(error, httpx2.TransportError)


def wait_time(error: httpx2.HTTPError, default: float = 0.5) -> float:
    """Seconds to wait before retrying: the server's Retry-After if it sent one."""
    if isinstance(error, httpx2.HTTPStatusError):
        retry_after = error.response.headers.get("Retry-After", "")
        if retry_after.isdigit():  # it can also be an HTTP date; we only handle seconds here
            return float(retry_after)
    return default


def describe(error: httpx2.HTTPError) -> str:
    if isinstance(error, httpx2.HTTPStatusError):
        return f"HTTPStatusError {error.response.status_code}"
    return f"{type(error).__name__}: {error}"


def get_with_one_retry(client: httpx2.Client, path: str) -> httpx2.Response:
    """GET, and if the failure is temporary, wait and try exactly once more.

    GET is safe to repeat. Be careful retrying POST: if the first request did reach the server,
    repeating it could create a second resource (or charge a second LLM call).
    """
    try:
        return client.get(path).raise_for_status()  # raise_for_status() returns the response
    except httpx2.HTTPError as error:  # base class of both HTTPStatusError and TransportError
        if not is_retryable(error):
            raise  # retrying can't help: let the caller deal with it
        delay = wait_time(error)
        print(f"   {describe(error)} -> retryable, waiting {delay:.1f}s")
        time.sleep(delay)
        return client.get(path).raise_for_status()


def free_local_port() -> int:
    """A port number on this machine where (almost certainly) nothing is listening."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


if __name__ == "__main__":
    # Timeouts can be set per phase: here at most 3 s to connect, 10 s for everything else
    # (each read, each write, waiting for a pooled connection).
    timeout = httpx2.Timeout(10.0, connect=3.0)
    print("timeout config:", timeout)

    fake_server = httpx2.MockTransport(flaky_api)
    client = httpx2.Client(base_url="https://flaky.example", timeout=timeout, transport=fake_server)
    with client:  # closes the connection pool when the block ends
        for path in ["/overloaded", "/rate-limited", "/slow", "/private"]:
            print(f"GET {path}")
            try:
                response = get_with_one_retry(client, path)
                print("   success:", response.json())
            except httpx2.HTTPError as error:
                print(f"   {describe(error)} -> NOT retryable: fix the request (here: the key)")

    print("GET a real local port with nothing listening")
    try:
        httpx2.get(f"http://127.0.0.1:{free_local_port()}/", timeout=3)
    except httpx2.ConnectError as error:  # a TransportError: we never got a response
        print(f"   {describe(error)}")
        print("   is_retryable:", is_retryable(error))
