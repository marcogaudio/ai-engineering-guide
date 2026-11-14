"""The same patterns with the real Anthropic SDK.

Needs an API key:  export ANTHROPIC_API_KEY=...
Run:               uv run chapters/01-python-foundations/examples/llm_client_demo.py
Without a key it prints a message and exits, so the tests still pass.
"""

import asyncio
import os
from typing import Literal

from anthropic import Anthropic, AsyncAnthropic
from pydantic import BaseModel, Field

MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5")


class Sentiment(BaseModel):
    label: Literal["positive", "negative", "neutral"]
    score: float = Field(ge=0, le=1)


def structured_output(client: Anthropic) -> Sentiment | None:
    """Pydantic model in, validated object out."""
    msg = client.messages.parse(
        model=MODEL,
        max_tokens=256,
        messages=[{"role": "user", "content": "Classify: 'The support team was fantastic'"}],
        output_format=Sentiment,
    )
    return msg.parsed_output


def stream_answer(client: Anthropic) -> None:
    """Print tokens as they arrive."""
    with client.messages.stream(
        model=MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": "Explain RAG in 3 sentences."}],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
        final = stream.get_final_message()
    print(f"\n[output tokens: {final.usage.output_tokens}]")


async def concurrent_eval(questions: list[str]) -> list[str | BaseException]:
    """Run many calls concurrently, at most 5 at a time."""
    client = AsyncAnthropic()
    sem = asyncio.Semaphore(5)

    async def ask(question: str) -> str:
        async with sem:
            msg = await client.messages.create(
                model=MODEL,
                max_tokens=100,
                messages=[{"role": "user", "content": question}],
            )
            block = msg.content[0]
            return block.text if block.type == "text" else ""

    return await asyncio.gather(*(ask(q) for q in questions), return_exceptions=True)


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set: skipping the live demo.")
        raise SystemExit(0)

    client = Anthropic()
    print("structured:", structured_output(client))
    stream_answer(client)
    answers = asyncio.run(concurrent_eval(["Capital of Italy?", "2 + 2?", "Largest planet?"]))
    print("concurrent:", answers)
