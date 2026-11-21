"""An LLM call without the SDK: just HTTP + JSON + an API key read from the environment.

This is what `client.chat.completions.create(...)` from the `openai` SDK does under the hood.
Same configuration as chapter 01's llm_client_demo.py. It defaults to a free local Ollama model:
    ollama pull llama3.2 && ollama serve
Or point it at any OpenAI-compatible API with LLM_BASE_URL, LLM_API_KEY and LLM_MODEL, set in
the environment or in a `.env` file in the repo root (copy `.env.example`).
Without a reachable server it prints a message and exits, so the tests still pass.

Run: uv run chapters/03-rest-apis-and-secrets/examples/07_raw_llm_call.py
"""

import json
from collections.abc import Iterator

import httpx2
from pydantic import BaseModel, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


# --- Configuration: from the environment or .env, never hard-coded ----------------------------
class LLMSettings(BaseSettings):
    # Reads LLM_BASE_URL, LLM_API_KEY and LLM_MODEL from the environment, then from `.env` in
    # the current folder (if it exists). `extra="ignore"`: other variables in .env are fine.
    model_config = SettingsConfigDict(env_prefix="LLM_", env_file=".env", extra="ignore")

    base_url: str = "http://localhost:11434/v1"
    api_key: SecretStr = SecretStr("ollama")  # Ollama ignores it; real providers require it
    model: str = "llama3.2"


# --- The response shape: only the fields we use ----------------------------------------------
class Message(BaseModel):
    role: str
    content: str | None = None  # None when the model answers with a tool call instead


class Choice(BaseModel):
    message: Message
    finish_reason: str | None = None  # "stop" = finished, "length" = hit max_tokens


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int


class ChatCompletion(BaseModel):
    model: str
    choices: list[Choice]
    usage: Usage | None = None


def make_client(settings: LLMSettings) -> httpx2.Client:
    return httpx2.Client(
        base_url=settings.base_url,
        # The key travels in a header, never in the URL. Unwrap the SecretStr only here.
        headers={"Authorization": f"Bearer {settings.api_key.get_secret_value()}"},
        # Generation can be slow (especially on a local model): long read, short connect.
        timeout=httpx2.Timeout(120.0, connect=5.0),
    )


def chat(client: httpx2.Client, model: str, prompt: str) -> ChatCompletion:
    """One request, one complete answer."""
    body = {
        "model": model,
        "max_tokens": 80,
        "messages": [{"role": "user", "content": prompt}],
    }
    print("POST /chat/completions with JSON body:", json.dumps(body))
    response = client.post("/chat/completions", json=body)
    response.raise_for_status()
    return ChatCompletion.model_validate(response.json())


def stream_chat(client: httpx2.Client, model: str, prompt: str) -> Iterator[str]:
    """The same call with "stream": true. The answer arrives as server-sent events (SSE)."""
    body = {
        "model": model,
        "max_tokens": 80,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
    }
    # client.stream() returns as soon as the headers arrive and lets us read the body piece by
    # piece, instead of waiting for the whole answer like client.post() does.
    with client.stream("POST", "/chat/completions", json=body) as response:
        response.raise_for_status()
        print("streaming, content-type:", response.headers["content-type"])
        shown_raw = False
        # SSE is plain text: each event is a line "data: <payload>", followed by an empty line.
        for line in response.iter_lines():
            if not line.startswith("data:"):
                continue  # the empty separator lines (and other SSE fields we don't use)
            data = line.removeprefix("data:").strip()
            if not shown_raw:
                print("first raw event:", line[:110] + "…")
                shown_raw = True
            if data == "[DONE]":  # OpenAI-style end marker: it is not JSON, so check it first
                break
            chunk = json.loads(data)
            # Each chunk carries a `delta`: only the new piece of text.
            for choice in chunk.get("choices", []):
                if text := choice["delta"].get("content"):
                    yield text


if __name__ == "__main__":
    settings = LLMSettings()
    print("settings:", settings)  # safe to print: the key shows as '**********'

    with make_client(settings) as client:
        # Health check: a cheap GET that lists the server's models. Short timeout, no retries,
        # so a missing server fails fast instead of hanging.
        try:
            client.get("/models", timeout=5).raise_for_status()
        except (httpx2.ConnectError, httpx2.TimeoutException):
            # Exit with code 0 (success): no server is an expected situation, not a bug, and the
            # test suite runs every example. `from None` hides the original traceback.
            print(f"No LLM server reachable at {settings.base_url}: skipping the live demo.")
            raise SystemExit(0) from None

        completion = chat(client, settings.model, "In one sentence: what is a REST API?")
        choice = completion.choices[0]
        print("answer:", choice.message.content)
        print("finish_reason:", choice.finish_reason, "| usage:", completion.usage)
        print()

        for text in stream_chat(client, settings.model, "Name three HTTP methods, briefly."):
            print(text, end="", flush=True)
        print()
