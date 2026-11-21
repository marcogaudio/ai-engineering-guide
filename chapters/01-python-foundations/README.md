# 01 · Python foundations for AI engineering

The five Python features you meet in almost every piece of LLM code. Each one gets the idea in one paragraph, a short snippet, and why it matters for LLM work.

| Topic | Why it matters for LLM work | Runnable example |
|---|---|---|
| [Type hints](#1-type-hints) | Readable client code; Pydantic and FastAPI read them to build schemas | [`type_hints.py`](examples/type_hints.py) |
| [Dataclasses vs Pydantic](#2-dataclasses-vs-pydantic) | Validate structured LLM output; generate tool schemas | [`pydantic_models.py`](examples/pydantic_models.py) |
| [Virtual environments](#3-virtual-environments-venv-and-uv) | Reproducible, isolated projects | — |
| [async/await](#4-asyncawait-for-llm-calls) | LLM calls are network-bound: run them concurrently | [`async_llm_calls.py`](examples/async_llm_calls.py) |
| [Generators & streaming](#5-generators-and-streaming) | Tokens arrive incrementally | [`streaming.py`](examples/streaming.py) |
| All of the above with a real LLM | | [`llm_client_demo.py`](examples/llm_client_demo.py) (free local model via Ollama, or any OpenAI-compatible API) |

---

## 1. Type hints

Type hints are labels that say what type each argument should be and what the function returns. **Python ignores them at runtime.** Your IDE and checkers like `mypy` or `pyright` use them to catch mistakes. Libraries like Pydantic and FastAPI also *read* them to build validation and JSON schemas, which is why they matter so much in LLM code.

```python
from typing import Literal

def embed(text: str) -> list[float]: ...

def search(
    query: str,
    top_k: int = 5,
    source: Literal["web", "docs"] = "docs",   # only these two strings are valid
    filters: dict[str, str] | None = None,     # optional: may be None
) -> list[dict[str, str]]: ...
```

Types you'll use most: `list[float]`, `dict[str, Any]`, `X | None`, `Literal[...]` for a fixed set of choices, `Callable[[str, str], bool]` for functions passed as arguments, and `Iterator[str]` / `AsyncIterator[str]` for streams.

Hints alone don't stop a bad call:

```python
def add(a: int, b: int) -> int:
    return a + b

add("4", "2")   # runs and returns "42"
# mypy: error: Argument 1 to "add" has incompatible type "str"; expected "int"
```

→ Run a type checker during development (`uv run mypy .`). When you need types enforced **at runtime**, use Pydantic.

## 2. Dataclasses vs Pydantic

Both define the shape of your data as a class with typed fields. The difference is what happens when data doesn't match. **A dataclass is just a container and accepts anything. Pydantic checks the data, converts it where it can, and generates a JSON schema.**

```python
from dataclasses import dataclass

@dataclass
class EvalResult:
    case_id: int
    output: str
    passed: bool

EvalResult(case_id=1, output="Paris", passed="yes")   # accepted without complaint
```

→ Use **dataclasses** for data your own code creates, such as configs and eval results.
→ Use **Pydantic** for anything that comes from outside: LLM output, API requests, user input.

```python
from typing import Literal
from pydantic import BaseModel, Field, ValidationError

class Sentiment(BaseModel):
    label: Literal["positive", "negative", "neutral"]
    score: float = Field(ge=0, le=1, description="Confidence between 0 and 1")
    reasons: list[str] = []

s = Sentiment.model_validate_json('{"label": "positive", "score": "0.92"}')
s.score    # 0.92 as a float (the string was converted)

try:
    Sentiment.model_validate_json('{"label": "happy", "score": 1.7}')
except ValidationError as e:
    print(e.errors())   # label isn't an allowed value, score > 1
                        # → send the errors back to the LLM and ask it to retry
```

**Pydantic has three main uses in LLM work:**

**a) Tool schemas.** `model_json_schema()` generates the JSON Schema that a tool definition needs:

```python
class SearchArgs(BaseModel):
    """Search the internal documentation."""
    query: str = Field(description="What to search for")
    top_k: int = Field(default=5, ge=1, le=20)

tool = {
    "name": "search_docs",
    "description": SearchArgs.__doc__,
    "input_schema": SearchArgs.model_json_schema(),
}
# When the model calls the tool, validate its arguments:
args = SearchArgs.model_validate(tool_use_block.input)
```

**b) Validating LLM output.** Parse the response, catch `ValidationError`, and send the error back to the model so it can fix its output.

**c) Structured output in the SDK.** You can pass the model class directly:

```python
msg = client.messages.parse(
    model="claude-sonnet-5-5",
    max_tokens=256,
    messages=[{"role": "user", "content": "Classify: 'The support team was fantastic'"}],
    output_format=Sentiment,
)
result = msg.parsed_output    # a Sentiment instance
```

The OpenAI SDK has an equivalent `.parse()` method.

## 3. Virtual environments: venv and uv

A virtual environment gives each project its own interpreter and its own installed packages. Upgrading a library for project A then can't break project B, and you can recreate the exact setup on another machine.

**The classic way:**

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install anthropic pydantic
pip freeze > requirements.txt      # records the exact versions installed
deactivate
```

**uv** is a much faster replacement written in Rust. It also manages Python versions and lockfiles:

```bash
uv init my-agent && cd my-agent    # creates pyproject.toml, .python-version, main.py, .gitignore, a git repo
uv add anthropic pydantic          # creates .venv and uv.lock automatically
uv add --dev pytest mypy           # development-only tools
uv run main.py                     # runs inside the env, no activation needed
uv sync                            # rebuilds the exact env from uv.lock on another machine
```

It also works as a drop-in replacement for pip: `uv venv` followed by `uv pip install -r requirements.txt`.

**Rules:** use one environment per project, never commit `.venv`, and always commit `uv.lock` or `requirements.txt`. This repo is itself a uv project: see the [root README](../../README.md#quick-start).

## 4. async/await for LLM calls

An LLM call spends almost all of its time waiting on the network, which is exactly the kind of work asyncio speeds up. Output of [`async_llm_calls.py`](examples/async_llm_calls.py), which simulates 50 calls of 1 second each:

| Strategy | Time |
|---|---|
| sequential (`await` one at a time) | ~50 s |
| `asyncio.gather` on all 50 | **1.00 s** |
| `gather` + `Semaphore(10)` | 5.01 s |

The same pattern with the real client:

```python
import asyncio
from anthropic import AsyncAnthropic

client = AsyncAnthropic()             # reads ANTHROPIC_API_KEY from the environment
sem = asyncio.Semaphore(10)           # at most 10 requests at once, to stay under rate limits

async def ask(question: str) -> str:
    async with sem:
        msg = await client.messages.create(
            model="claude-sonnet-5-5",
            max_tokens=512,
            messages=[{"role": "user", "content": question}],
        )
        return msg.content[0].text

async def run_eval(questions: list[str]) -> list[str | BaseException]:
    return await asyncio.gather(*(ask(q) for q in questions), return_exceptions=True)

results = asyncio.run(run_eval(questions))   # in Jupyter: results = await run_eval(questions)
```

Three things to get right:

- **Use the async client** (`AsyncAnthropic`, `AsyncOpenAI`). A sync client inside `async def` blocks the event loop, and your calls run one at a time again.
- **Use a Semaphore.** Sending 500 requests at once will hit rate limits (HTTP 429 errors).
- **Pass `return_exceptions=True`.** One failed example then doesn't crash the whole eval. Results come back in input order, so you can `zip(questions, results)`.

Want the full picture of how asyncio works? See [Chapter 02](../02-asyncio/README.md).

## 5. Generators and streaming

A function that contains `yield` is a **generator**. It produces values one at a time and pauses between them, and none of its code runs until you start iterating.

```python
import time
from collections.abc import Iterator

def fake_stream(text: str) -> Iterator[str]:
    for word in text.split():
        time.sleep(0.05)          # simulates tokens arriving over the network
        yield word + " "

gen = fake_stream("The answer is forty two")   # <generator object ...>: nothing has run yet
next(gen)                                       # 'The '

for chunk in fake_stream("The answer is forty two"):
    print(chunk, end="", flush=True)            # text appears word by word
```

**Why stream LLM output:** the user sees the first tokens almost immediately instead of waiting several seconds for the whole answer.

```python
from anthropic import Anthropic
client = Anthropic()

with client.messages.stream(
    model="claude-sonnet-5-5",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Explain RAG in 3 sentences"}],
) as stream:
    for text in stream.text_stream:          # yields each piece of text as it arrives
        print(text, end="", flush=True)
    final = stream.get_final_message()       # the full message, including token usage
```

- **Async version:** use `AsyncAnthropic`, `async with client.messages.stream(...)` and `async for text in stream.text_stream`. This is what you'd use in a FastAPI server.
- **OpenAI style:** pass `stream=True`, then loop `for chunk in stream:` and read `chunk.choices[0].delta.content`.

Generators are useful beyond streaming. They let you process large eval datasets one record at a time, so memory use stays flat:

```python
def read_jsonl(path):
    with open(path) as f:
        for line in f:
            yield json.loads(line)           # only one record in memory at a time

def failures(records):
    for r in records:
        if not r["passed"]:
            yield r

for r in failures(read_jsonl("eval_results.jsonl")):
    print(r["id"])
```

---

**How they fit together:** a typical eval script uses all five. You set up the project with uv, define test cases and grader output as Pydantic models, and run the calls concurrently with the async client and `gather`. The user-facing app streams its responses, and type hints run through all of it.

**Next:** [02 · asyncio in depth →](../02-asyncio/README.md)
