"""JSON: the text format that almost every REST API, and every LLM API, speaks.

Run: uv run chapters/03-rest-apis-and-secrets/examples/02_json.py
"""

import json
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ValidationError


# --- 4. The shape we expect from an API response, as a Pydantic model -----------------------
class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class Reply(BaseModel):
    id: str
    model: str
    text: str
    stop_reason: Literal["end_turn", "max_tokens", "tool_use"]  # only these three are valid
    usage: Usage  # a nested JSON object becomes a nested model


def section(title: str) -> None:
    print(f"\n--- {title} " + "-" * (60 - len(title)))


if __name__ == "__main__":
    # --- 1. Python -> JSON text -> Python ------------------------------------------------------
    section("1. round trip")
    request_body = {
        "model": "llama3.2",
        "max_tokens": 256,  # int    -> number
        "temperature": 0.2,  # float  -> number
        "stream": False,  # bool   -> false (lowercase in JSON)
        "stop": None,  # None   -> null
        "tags": ("demo", "ch03"),  # tuple  -> array (comes back as a list!)
        "messages": [{"role": "user", "content": "Hi"}],  # list/dict -> array/object
    }
    # dumps = "dump to string": Python object -> JSON text. This is what goes in a request body.
    text = json.dumps(request_body)
    print("json.dumps:", text)
    # loads = "load from string": JSON text -> Python object. This is what you do with a response.
    back = json.loads(text)
    print("json.loads:", back)
    print("tags came back as:", type(back["tags"]).__name__)  # list, not tuple
    # indent=2 makes it readable for humans (logs, debugging); APIs don't care either way.
    print("pretty:\n" + json.dumps(back["messages"], indent=2))

    # --- 2. Pitfalls, each with its fix --------------------------------------------------------
    section("2. pitfalls")
    created = datetime(2026, 11, 21, 9, 30, tzinfo=UTC)
    try:
        json.dumps({"created": created})
    except TypeError as e:
        print("datetime     ->", e)  # JSON has no date type
    print("  fix        ->", json.dumps({"created": created.isoformat()}))  # send a string

    print("int keys     ->", json.loads(json.dumps({1: "a", 2: "b"})))  # keys become strings

    print("NaN          ->", json.dumps({"score": float("nan")}))  # 'NaN' is NOT valid JSON
    try:
        json.dumps({"score": float("nan")}, allow_nan=False)  # make it fail loudly instead
    except ValueError as e:
        print("  fix        ->", e)

    print("non-ASCII    ->", json.dumps({"city": "Napoli, caffè ☕"}))  # escaped by default
    print("  fix        ->", json.dumps({"city": "Napoli, caffè ☕"}, ensure_ascii=False))

    # LLMs often wrap JSON in prose or markdown, or leave a trailing comma: loads() refuses it.
    for bad in ['{"label": "positive",}', '```json\n{"label": "positive"}\n```']:
        try:
            json.loads(bad)
        except json.JSONDecodeError as e:
            print(f"invalid JSON -> {e.msg} at line {e.lineno} col {e.colno}: {bad!r}")

    # --- 3. JSONL: one JSON object per line (datasets, eval results, batch API files) -------
    section("3. JSONL")
    records = [{"id": 1, "passed": True}, {"id": 2, "passed": False}]
    jsonl = "\n".join(json.dumps(r) for r in records)  # what you'd write to results.jsonl
    print(jsonl)
    print("failed ids:", [r["id"] for r in map(json.loads, jsonl.splitlines()) if not r["passed"]])

    # --- 4. Validate what comes back with Pydantic -------------------------------------------
    section("4. validate with Pydantic")
    good = (
        '{"id": "msg_01", "model": "demo", "text": "Hello!", "stop_reason": "end_turn",'
        ' "usage": {"input_tokens": "12", "output_tokens": 3}, "extra_field": "ignored"}'
    )
    # One call: parse the JSON text AND check every field's type. "12" is converted to 12,
    # unknown fields are ignored. You then get attribute access with IDE autocompletion.
    reply = Reply.model_validate_json(good)
    print("valid  ->", reply.text, "| output tokens:", reply.usage.output_tokens)

    bad = '{"id": "msg_02", "model": "demo", "stop_reason": "stop", "usage": {"input_tokens": 1}}'
    try:
        Reply.model_validate_json(bad)
    except ValidationError as e:
        # Every problem at once, with its location: far better than a KeyError three calls later.
        for err in e.errors():
            print("invalid ->", ".".join(map(str, err["loc"])), ":", err["msg"])

    # And back to JSON text, e.g. to store the reply or return it from your own API.
    print("model_dump_json ->", reply.model_dump_json())
