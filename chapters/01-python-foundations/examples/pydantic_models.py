"""Dataclasses vs Pydantic: model request/response shapes and validate LLM output.

Run: uv run chapters/01-python-foundations/examples/pydantic_models.py
"""

import json
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

# --- 1. Dataclass: a typed container, NO validation -------------------------


@dataclass
class EvalResult:
    case_id: int
    output: str
    passed: bool
    latency_s: float = 0.0


# --- 2. Pydantic: validates, converts types, and generates JSON Schema ------


class Sentiment(BaseModel):
    label: Literal["positive", "negative", "neutral"]
    score: float = Field(ge=0, le=1, description="Confidence between 0 and 1")
    reasons: list[str] = []


# --- 3. Tool schema generated from a Pydantic model -------------------------


class SearchArgs(BaseModel):
    """Search the internal documentation."""

    query: str = Field(description="What to search for")
    top_k: int = Field(default=5, ge=1, le=20)


def tool_from_model(name: str, model: type[BaseModel]) -> dict:
    """Build a tool definition in the shape LLM APIs expect."""
    return {
        "name": name,
        "description": (model.__doc__ or "").strip(),
        "input_schema": model.model_json_schema(),
    }


def parse_llm_output(raw: str) -> Sentiment | list[str]:
    """Return a Sentiment, or the list of errors to send back to the LLM for a retry."""
    try:
        return Sentiment.model_validate_json(raw)
    except ValidationError as e:
        return [f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()]


if __name__ == "__main__":
    # A dataclass accepts wrong types without complaint
    r = EvalResult(case_id=1, output="Paris", passed="yes")  # type: ignore[arg-type]
    print("dataclass accepted:", r)

    # Pydantic: valid JSON (score arrives as a string and is converted to float)
    good = parse_llm_output('{"label": "positive", "score": "0.92", "reasons": ["fast"]}')
    print("valid  ->", good)

    # Pydantic: invalid JSON gives readable errors you can feed back to the LLM
    bad = parse_llm_output('{"label": "happy", "score": 1.7}')
    print("invalid ->", bad)

    # Tool definition ready to pass to an LLM API
    tool = tool_from_model("search_docs", SearchArgs)
    print("tool schema:", json.dumps(tool, indent=2))

    # When the model calls the tool, validate its arguments the same way
    args = SearchArgs.model_validate({"query": "refund policy", "top_k": 3})
    print("tool args:", args)
