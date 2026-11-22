# 03 · REST APIs, JSON and secrets

Every LLM provider, vector database and tool your agent calls is reached through an HTTP API that sends JSON back and forth and checks an API key. This chapter covers how HTTP works, how to call REST APIs from Python and handle what comes back, and how to keep your keys out of your code and out of git.

| Topic | Why it matters for LLM work | Runnable example |
|---|---|---|
| [How HTTP works](#1-how-http-works) | Every SDK call is an HTTP request: you can debug it when you can read it | [`01_http_by_hand.py`](examples/01_http_by_hand.py) |
| [Anatomy of a request and a response](#2-anatomy-of-a-request-and-a-response) | Headers carry your key, the body carries your prompt | [`01_http_by_hand.py`](examples/01_http_by_hand.py) |
| [REST: resources, methods and URLs](#3-rest-resources-methods-and-urls) | The shape of almost every API you'll integrate | [`03_rest_client.py`](examples/03_rest_client.py) |
| [Status codes](#4-status-codes) | Tell "retry later" apart from "fix your request" | [`03_rest_client.py`](examples/03_rest_client.py) |
| [JSON](#5-json) | Request bodies, responses, tool arguments, eval datasets | [`02_json.py`](examples/02_json.py) |
| [Calling APIs with httpx2](#6-calling-apis-with-httpx2) | The client library under the `openai` and `anthropic` SDKs | [`03_rest_client.py`](examples/03_rest_client.py) |
| [Errors, timeouts and retries](#7-errors-timeouts-and-retryable-responses) | Networks fail; LLM APIs are often overloaded | [`04_errors_and_timeouts.py`](examples/04_errors_and_timeouts.py) |
| [Async calls to a real API](#8-async-calls-to-a-real-api) | Many calls at once, with chapter 02's tools | [`05_async_public_api.py`](examples/05_async_public_api.py) |
| [API keys and environment variables](#9-api-keys-and-environment-variables) | Config and secrets live outside the code | [`06_env_and_secrets.py`](examples/06_env_and_secrets.py) |
| [Secret hygiene](#10-secret-hygiene) | A leaked LLM key is someone else's free GPU budget | — |
| [Putting it together](#11-putting-it-together-an-llm-call-without-the-sdk) | An LLM call with no SDK: just HTTP, JSON and a key | [`07_raw_llm_call.py`](examples/07_raw_llm_call.py) (free local model via Ollama, or any OpenAI-compatible API) |

![The life of one API call, mapped to this chapter's sections: config and secrets (§9–10) feed one httpx2 client (§6), which sends a request (§2, §3, §5) over HTTP (§1). A 2xx response (§4) is validated with Pydantic (§5); an error status or no response goes to §7, which retries through the client or fixes the request.](images/chapter-map.svg)

---

## 1. How HTTP works

**HTTP** (HyperText Transfer Protocol) is the protocol the web runs on. It's a **client-server** protocol: the client always starts the exchange by sending a **request**, and the server answers with a **response**. The client is called the **user agent**. It's usually a browser, but your Python script is one too.

Between the two there are often **proxies**: machines that forward the messages, and on the way may cache them, filter them, balance load across servers, or log them. Keep the logging in mind for section 9.

Four properties explain most of HTTP's design:

- **Simple.** Up to HTTP/1.1 the messages are human-readable text. HTTP/2 packs them into binary frames, but the meaning is unchanged.
- **Extensible.** **Headers** (`Name: value` lines) can carry any extra information, so new features such as caching, authentication and API keys were added as new headers rather than new protocols.
- **Stateless.** The server doesn't link one request to the next, so every request must carry everything the server needs. Sessions are built on top, with cookies in browsers or a key sent on every API call.
- **It needs a reliable connection,** which in practice means TCP. HTTP/1.0 opened a new TCP connection for every request. HTTP/1.1 keeps connections open and reuses them (**persistent connections**). HTTP/2 sends many requests at once over a single connection (**multiplexing**). HTTP/3 moves to QUIC, which is built on UDP.

![Top: your Python script talks to the server, possibly through a proxy, in four steps: open a TCP connection, send the request, read the response, then close the connection or reuse it. Bottom: three requests over time. HTTP/1.0 opens a new connection for each, HTTP/1.1 reuses one connection for requests one after another, and HTTP/2 sends all three at once over one connection and finishes first.](images/http-flow.svg)

**The HTTP flow** is four steps: open a TCP connection, send a request, read the response, then close the connection or reuse it. [`01_http_by_hand.py`](examples/01_http_by_hand.py) does exactly this over a raw socket against a tiny local server, so you can read the actual text. Here is what it sends and receives:

```http
GET /hello HTTP/1.1
Host: 127.0.0.1:49978
Accept: application/json
Connection: close

HTTP/1.1 200 OK
Server: BaseHTTP/0.6 Python/3.12.16
Date: Sat, 21 Nov 2026 16:05:07 GMT
Content-Type: application/json
Content-Length: 54

{"message": "hello from the server", "path": "/hello"}
```

**In LLM work:**
- `client.messages.create(...)` is an HTTPS POST with a JSON body. When something breaks, think in terms of the request you sent and the response you got back.
- **LLM APIs are stateless too.** The server doesn't remember your conversation, which is why you send the *whole* `messages` list on every turn, and why long chats cost more and more input tokens.

## 2. Anatomy of a request and a response

```text
POST /v1/chat/completions HTTP/1.1          ← method · path (+ ?query) · version
Host: api.example.com                       ┐
Authorization: Bearer sk-...                │ headers
Content-Type: application/json              ┘
                                            ← empty line
{"model": "llama3.2", "messages": [...]}    ← body (optional: GET has none)

HTTP/1.1 200 OK                             ← version · status code · reason phrase
Content-Type: application/json              ← headers
                                            ← empty line
{"choices": [...], "usage": {...}}          ← body
```

A **URL** like `https://api.example.com/v1/notes?limit=10` breaks down into the scheme (`https`), the host (`api.example.com`), the path (`/v1/notes`) and the query string (`limit=10`). The request line carries only the path and query string; the host goes in the `Host` header.

![A URL split into scheme, host, path and query string, showing the path and query going into the request line and the host into the Host header. Below, a POST request and its 201 Created response, part by part (start line, headers, empty line, body), each labelled with the httpx2 attribute that exposes it, such as r.request.headers, r.status_code, r.headers and r.json().](images/message-anatomy.svg)

| Header | Sent by | Meaning |
|---|---|---|
| `Content-Type` | both | Format of the body: `application/json`, `text/event-stream` (streaming), … |
| `Accept` | client | Formats the client wants back |
| `Authorization` | client | Credentials, e.g. `Bearer <api key>` |
| `User-Agent` | client | Which program is calling |
| `Content-Length` | both | Size of the body in bytes |
| `Retry-After` | server | With 429/503: how many seconds to wait before retrying |

## 3. REST: resources, methods and URLs

**REST** is a style of API design, not a protocol: it's a set of conventions on top of HTTP. The API exposes **resources** (notes, files, models, batches) at URLs, and the HTTP **method** says what to do with them:

| Method | Does | Safe? | Idempotent? | Typical success |
|---|---|---|---|---|
| `GET /notes` · `GET /notes/42` | read the collection, or one item | ✅ | ✅ | `200 OK` |
| `POST /notes` | create a new item | ❌ | ❌ | `201 Created` (+ `Location` header) |
| `PUT /notes/42` | replace an item entirely | ❌ | ✅ | `200 OK` / `204` |
| `PATCH /notes/42` | change some fields | ❌ | ❌ (usually) | `200 OK` |
| `DELETE /notes/42` | delete an item | ❌ | ✅ | `204 No Content` |

- **Safe** means the request doesn't change anything on the server.
- **Idempotent** means sending it twice has the same effect as sending it once. That's what makes a request **safe to retry** (section 7).

Conventions you'll see everywhere:
- plural nouns for collections (`/notes`), with an id for a single item (`/notes/42`);
- query parameters to filter and paginate (`?limit=20&after=note_41`);
- a version in the path (`/v1/...`).

**In LLM work:** creating a completion is a `POST` and is **not idempotent**. Retrying one that reached the server means paying for it twice. Provider APIs also have plain REST resources: `GET /v1/models`, plus files and batches you can create, read and delete.

## 4. Status codes

The first digit gives the class: **1xx** informational · **2xx** success · **3xx** redirection · **4xx** *you* made a mistake · **5xx** the *server* had a problem.

| Code | Meaning | What your code should do |
|---|---|---|
| 200 / 201 / 204 | OK / Created / No Content | carry on |
| 400 Bad Request | malformed request (e.g. invalid parameter) | fix the request |
| 401 Unauthorized | missing or invalid API key | fix the key, don't retry |
| 403 Forbidden | valid key, but no permission | fix the permissions |
| 404 Not Found | wrong URL or the resource doesn't exist | fix the URL or id |
| 413 Content Too Large | body too big (e.g. an enormous prompt or file) | send less |
| 422 Unprocessable Content | valid JSON, but the fields are wrong | fix the body |
| **429 Too Many Requests** | rate limited | **wait (`Retry-After`), then retry** |
| **500 / 502 / 503 / 504** | server error, bad gateway, overloaded, gateway timeout | **wait, then retry** |

Rule of thumb: **4xx means fix the request, 5xx and 429 mean retry later.** Some providers add their own codes. Anthropic, for example, returns `529` when its API is overloaded, which you should treat like a 503.

Section 7 combines these rules with network errors into [one decision tree](#7-errors-timeouts-and-retryable-responses).

## 5. JSON

JSON is text that represents nested objects and arrays. It's the body format of nearly every REST API and every LLM API, and the format of tool-call arguments and eval datasets (JSONL).

| JSON | Python |
|---|---|
| object `{"k": 1}` | `dict` |
| array `[1, 2]` | `list` (a `tuple` is sent as an array and comes back as a `list`) |
| string `"hi"` | `str` |
| number `1` / `1.5` | `int` / `float` |
| `true` / `false` | `True` / `False` |
| `null` | `None` |

```python
import json

text = json.dumps({"model": "llama3.2", "stream": False, "stop": None})  # Python -> JSON text
data = json.loads('{"usage": {"output_tokens": 3}}')                     # JSON text -> Python
data["usage"]["output_tokens"]   # 3
```

Pitfalls, all shown in [`02_json.py`](examples/02_json.py):
- `datetime` isn't serializable: send `dt.isoformat()`.
- Dict keys always come back as strings (`{1: "a"}` → `{"1": "a"}`).
- `json.dumps(float("nan"))` produces `NaN`, which is **not valid JSON**: pass `allow_nan=False` to catch it.
- Non-ASCII text is escaped by default: pass `ensure_ascii=False` for readable output.
- LLMs often wrap JSON in markdown fences or leave a trailing comma, and `json.loads` raises `JSONDecodeError`.

**Validate, don't just parse.** `json.loads` tells you the text is valid JSON. It doesn't tell you the fields you need are there. Parse and validate in one step with Pydantic (see [Chapter 01 §2](../01-python-foundations/README.md#2-dataclasses-vs-pydantic)):

```python
reply = Reply.model_validate_json(response.text)   # ValidationError lists every bad field at once
reply.usage.output_tokens                          # typed attribute access from here on
```

## 6. Calling APIs with httpx2

[`httpx2`](https://pydantic.dev/docs/httpx2/) is the maintained successor of `httpx`, and both the `openai` and `anthropic` SDKs use it under the hood. Its API looks like `requests`, but it also has an async client.

```python
import httpx2

with httpx2.Client(
    base_url="https://api.notes.example",
    headers={"Authorization": f"Bearer {api_key}"},   # sent with every request
    timeout=10,                                       # seconds: always set one
) as client:
    r = client.post("/notes", json={"text": "read MDN"})   # json= serializes + sets Content-Type
    r.status_code, r.headers["Location"], r.json()          # 201, '/notes/1', {...}

    r = client.get("/notes", params={"limit": 5})           # → GET /notes?limit=5
    r.raise_for_status()                                    # 4xx/5xx → HTTPStatusError
    notes = r.json()["data"]
```

- **Create one client and reuse it.** It keeps a pool of open connections: HTTP/1.1's persistent connections, from section 1. A bare `httpx2.get(...)` opens and closes a new connection every time.
- **httpx2 does not raise on 4xx/5xx by itself.** Check `r.is_success`, or call `r.raise_for_status()`.
- **Redirects aren't followed by default.** Pass `follow_redirects=True` if an API answers with 3xx.
- **The default timeout is 5 s.** That's fine for a REST API but too short for an LLM writing a long answer, which is why the SDKs default to several minutes.
- **`httpx2.MockTransport(handler)` swaps the network for a Python function.** That's how [`03_rest_client.py`](examples/03_rest_client.py) runs a complete CRUD session against a fake notes API offline, and how you unit-test your own API clients.

## 7. Errors, timeouts and retryable responses

Two different things can go wrong:

| Exception | What happened | Retry? |
|---|---|---|
| `httpx2.HTTPStatusError` (from `raise_for_status()`) | the server **answered** with 4xx/5xx | only 408, 429 and 5xx |
| `httpx2.TransportError`: `ConnectError`, `ReadTimeout`, … | **no usable answer**: refused, dropped, or too slow | yes, usually temporary |

Both inherit from `httpx2.HTTPError`.

![Decision tree for a failed call. No response at all is a TransportError (ConnectError, ReadTimeout) and is usually temporary. With a response, 2xx means carry on, 3xx is a redirect (follow_redirects=True), 408, 429, 500, 502, 503 and 504 mean wait (honour Retry-After) and retry, and any other error status means fix the request. Only idempotent requests are safe to retry blindly: a POST that timed out may already have run.](images/retry-decision.svg)

```python
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}

def is_retryable(error: httpx2.HTTPError) -> bool:
    if isinstance(error, httpx2.HTTPStatusError):
        return error.response.status_code in RETRYABLE_STATUS
    return isinstance(error, httpx2.TransportError)

timeout = httpx2.Timeout(60.0, connect=3.0)   # fail fast if the host is down, wait for slow answers
```

- **Honour `Retry-After`** when the server sends it: the server is telling you exactly how long to wait.
- **Only retry idempotent requests blindly.** A `POST` that timed out may still have been processed on the server.

[`04_errors_and_timeouts.py`](examples/04_errors_and_timeouts.py) shows each case (503, 429 with `Retry-After`, a read timeout, a 401, and a real refused connection) with a single retry. Retry loops with exponential backoff and jitter, and token-based rate limits, are covered in the next chapter.

## 8. Async calls to a real API

`httpx2.AsyncClient` takes the same arguments as `Client`, but its methods are coroutines. Combined with `gather` from [Chapter 02](../02-asyncio/README.md#9-create_task-gather-and-as_completed), many requests wait on the network at the same time:

```python
async def fetch_info(client: httpx2.AsyncClient, package: str) -> PackageInfo:
    r = await client.get(f"/pypi/{package}/json")
    r.raise_for_status()
    return PackageInfo.model_validate(r.json()["info"])

async def main() -> None:
    async with httpx2.AsyncClient(base_url="https://pypi.org", timeout=15) as client:
        infos = await asyncio.gather(*(fetch_info(client, p) for p in PACKAGES))
```

![Timeline of six requests. Run one after another, the total time is the sum of all six. With asyncio.gather on one AsyncClient, all six wait on the network at the same time, so the total is about as long as the slowest one. Python only works briefly at the start and end of each request.](images/sequential-vs-gather.svg)

[`05_async_public_api.py`](examples/05_async_public_api.py) fetches six packages from the PyPI JSON API in about 0.1 s. It also prints the real response headers, including `Cache-Control` and `ETag`: HTTP's caching controls in action.

## 9. API keys and environment variables

**Where the key goes:** in a **header**, never in the URL. URLs end up in proxy logs, server logs and browser history (section 1). Providers use `Authorization: Bearer <key>` (OpenAI-style) or a custom header like `x-api-key` (Anthropic).

**Where the key comes from:** the **environment**, never the source code. Every process inherits a set of `NAME=value` text variables from the shell that started it:

```bash
export LLM_API_KEY="sk-..."     # for this terminal session only
uv run app.py
```

```python
import os

api_key = os.environ["LLM_API_KEY"]               # required: KeyError right away if missing
model = os.getenv("LLM_MODEL", "llama3.2")        # optional: falls back to a default
```

**A `.env` file** saves you from exporting variables by hand in every new terminal. It's a file of `NAME=value` lines in the project folder:

```
.env           ← real values. Listed in .gitignore, NEVER committed
.env.example   ← same names, placeholder values. Committed, as documentation
```

Three ways to load it, from least to most code:

```bash
uv run --env-file .env app.py          # uv loads it, your code just reads os.environ
```

```python
from dotenv import load_dotenv          # python-dotenv
load_dotenv()                           # copies .env into os.environ (existing variables win)
```

```python
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LLM_", env_file=".env")
    api_key: SecretStr                       # required: the app won't start without it
    base_url: str = "http://localhost:11434/v1"
    timeout_s: float = 30.0                  # "10" in the environment becomes 10.0

settings = Settings()
print(settings)        # api_key=SecretStr('**********') base_url=... : safe to log
headers = {"Authorization": f"Bearer {settings.api_key.get_secret_value()}"}   # unwrap only here
```

`pydantic-settings` is the most robust of the three:
- it declares all config in one place, with types;
- it **fails at startup** with a clear message when a required value is missing;
- `SecretStr` keeps the key out of `print()`, logs and tracebacks.

The priority order is: real environment variables, then `.env`, then the defaults. Run [`06_env_and_secrets.py`](examples/06_env_and_secrets.py) to see each step with a fake key.

![Environment variables, then the .env file, then class defaults, in priority order, feed Settings(), which validates them and wraps the key in SecretStr so printing shows only asterisks. The key is unwrapped with .get_secret_value() only to build the Authorization header, never put in the URL. Below, .gitignore blocks .env but allows .env.example, so only the placeholder file is committed. Git keeps every commit forever, so a leaked key must be rotated.](images/config-and-secrets.svg)

**In LLM work:** the official SDKs read their key from a standard variable when you don't pass one: `Anthropic()` uses `ANTHROPIC_API_KEY`, and `OpenAI()` uses `OPENAI_API_KEY` (plus `OPENAI_BASE_URL`). This repo's examples use `LLM_BASE_URL`, `LLM_API_KEY` and `LLM_MODEL` so they work with any OpenAI-compatible provider: copy [`.env.example`](../../.env.example) to `.env` and fill it in.

## 10. Secret hygiene

- **Never hardcode a key**, not even "just for a quick test". Quick tests get committed.
- **Never commit `.env`.** Check that `.gitignore` covers it *before* the first `git add .`.
- **Never print or log a key.** Redact the `Authorization` / `x-api-key` headers before logging a request (see `redact()` in `06_env_and_secrets.py`).
- **Never ship a key to a browser or mobile app.** Anyone can read it there. Call the LLM from your backend.
- **If a key leaks, rotate it.** Revoke it in the provider's console and create a new one. Deleting the file isn't enough, because git history keeps every version and bots scan public repos for keys within minutes.
- **Use one key per project and environment** (dev, CI, prod), with the lowest permissions and spending limits the provider offers. Revoking one leaked key then doesn't break everything else.
- **In CI, use the platform's secret store.** Add the key under the repo's *Settings → Secrets and variables → Actions*, then pass it to a workflow step. GitHub masks the value in the logs:

  ```yaml
  env:
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
  ```
- **Scan for secrets automatically:** GitHub's secret scanning and push protection, or [gitleaks](https://github.com/gitleaks/gitleaks) as a pre-commit hook.
- **In production,** load secrets from a secret manager (AWS Secrets Manager, GCP Secret Manager, Vault, …) into environment variables. More in the Deployment chapter.

## 11. Putting it together: an LLM call without the SDK

An SDK call is just everything above combined: a POST, a JSON body, a key in a header, a JSON response. [`07_raw_llm_call.py`](examples/07_raw_llm_call.py) calls any OpenAI-compatible API with nothing but `httpx2` and Pydantic:

```python
settings = LLMSettings()                                        # LLM_* from the environment / .env
with httpx2.Client(
    base_url=settings.base_url,
    headers={"Authorization": f"Bearer {settings.api_key.get_secret_value()}"},
    timeout=httpx2.Timeout(120.0, connect=5.0),
) as client:
    body = {
        "model": settings.model,
        "max_tokens": 80,
        "messages": [{"role": "user", "content": "In one sentence: what is a REST API?"}],
    }
    r = client.post("/chat/completions", json=body)
    r.raise_for_status()
    completion = ChatCompletion.model_validate(r.json())
    print(completion.choices[0].message.content)
```

**Streaming uses server-sent events (SSE),** a standard way for a server to push a series of events over one HTTP response. With `"stream": true` the response has `Content-Type: text/event-stream`, and its body arrives line by line as it's generated:

```text
data: {"choices":[{"delta":{"content":"GET"}}], ...}

data: {"choices":[{"delta":{"content":":"}}], ...}

data: [DONE]
```

```python
with client.stream("POST", "/chat/completions", json=body | {"stream": True}) as r:
    for line in r.iter_lines():
        if line.startswith("data:") and (data := line[5:].strip()) != "[DONE]":
            print(json.loads(data)["choices"][0]["delta"].get("content") or "", end="")
```

![What 07_raw_llm_call.py does. LLMSettings reads LLM_BASE_URL, LLM_API_KEY and LLM_MODEL. An httpx2 client with the Bearer header and a timeout sends POST /chat/completions to the LLM server, and the 200 JSON response is validated into ChatCompletion to get the answer text. Streaming: the same POST with "stream": true returns data: lines, each with a small text delta, until data: [DONE], while the printed text grows.](images/raw-llm-call.svg)

The same call to Anthropic's Messages API differs only in the URL, the headers and the JSON shapes:

```python
httpx2.post(
    "https://api.anthropic.com/v1/messages",
    headers={
        "x-api-key": os.environ["ANTHROPIC_API_KEY"],
        "anthropic-version": "2023-06-01",
    },
    json={"model": "claude-sonnet-5-5", "max_tokens": 80, "messages": [...]},
    timeout=60,
)   # reply text in r.json()["content"][0]["text"]; streaming uses named SSE events
```

In practice, use the SDKs: they add retries, typed responses and streaming helpers. Knowing what's underneath is what lets you debug them, call a provider that has no SDK, and read their error messages.

---

**How they fit together:** your app reads its key and base URL from the environment with `pydantic-settings`. It creates one `httpx2` client (or the SDK's) with the key in a header and a sensible timeout. It sends JSON, validates the JSON that comes back with Pydantic, retries 429s and 5xx, and fixes 4xx. The key never appears in the code, the logs or git.

**Further reading:** [MDN, *Overview of HTTP*](https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/Overview) (the main source for this chapter) · [MDN, HTTP messages](https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/Messages) · [MDN, status codes](https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Status) · [MDN, request methods](https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Methods) · [MDN, using server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events) · [httpx2 docs](https://pydantic.dev/docs/httpx2/) · [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) · [The Twelve-Factor App: Config](https://12factor.net/config) · [OWASP Secrets Management Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html) · [Using secrets in GitHub Actions](https://docs.github.com/en/actions/security-for-github-actions/security-guides/using-secrets-in-github-actions)

**Previous:** [← 02 · asyncio in depth](../02-asyncio/README.md) · **Next:** 04 · Calling LLM APIs (coming soon)
