"""Type hints: document what goes in and out of a function.

Run:  uv run chapters/01-python-foundations/examples/type_hints.py
Check: uv run mypy chapters/01-python-foundations/examples/type_hints.py
"""

from collections.abc import Callable, Iterator
from typing import Any, Literal


# Basic: arguments and return value
def embed(text: str) -> list[float]:
    """Fake embedding: a real one would call an embeddings API."""
    return [float(len(text)), float(text.count(" "))]


# The types you'll use most in LLM code
def search(
    query: str,
    top_k: int = 5,
    source: Literal["web", "docs"] = "docs",  # only these two strings are valid
    filters: dict[str, str] | None = None,  # optional: may be None
) -> list[dict[str, Any]]:
    results = [{"query": query, "source": source, "rank": i} for i in range(top_k)]
    if filters:
        results = [r | {"filters": filters} for r in results]
    return results


# Functions as arguments: e.g. a grader passed to an eval loop
Grader = Callable[[str, str], bool]


def exact_match(output: str, expected: str) -> bool:
    return output.strip().lower() == expected.strip().lower()


def run_eval(cases: list[tuple[str, str]], grader: Grader) -> float:
    passed = sum(grader(out, exp) for out, exp in cases)
    return passed / len(cases)


# Generators get typed too (see streaming.py)
def tokens(text: str) -> Iterator[str]:
    yield from text.split()


if __name__ == "__main__":
    print("embed:", embed("hello type hints"))
    print("search:", search("refund policy", top_k=2, filters={"lang": "en"}))
    print("accuracy:", run_eval([("Paris", "paris"), ("Rome", "Milan")], exact_match))
    print("tokens:", list(tokens("hints are documentation")))

    # Annotations are stored on the function, and libraries like Pydantic read them
    print("annotations:", embed.__annotations__)

    # Hints are NOT enforced at runtime: this runs and returns "42".
    # `mypy` would report: Argument 1 to "add" has incompatible type "str"; expected "int"
    def add(a: int, b: int) -> int:
        return a + b

    print('add("4", "2") =', add("4", "2"))  # type: ignore[arg-type]
