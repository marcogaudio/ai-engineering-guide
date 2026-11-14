"""Handling several concurrent failures: ExceptionGroup and except* (Python 3.11+).

Run: uv run chapters/02-asyncio/examples/06_exception_groups.py
"""

import asyncio


async def boom(exc: Exception) -> None:
    await asyncio.sleep(0.1)
    raise exc


async def main() -> None:
    results = await asyncio.gather(
        boom(ValueError("bad value")),
        boom(TypeError("bad type")),
        boom(IndexError("bad index")),
        return_exceptions=True,  # errors come back as results instead of being raised
    )
    errors = [r for r in results if isinstance(r, Exception)]
    if errors:
        raise ExceptionGroup("several calls failed", errors)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except* ValueError as eg:
        print("handled ValueError:", eg.exceptions)
    except* (TypeError, IndexError) as eg:
        print("handled the others:", eg.exceptions)
