# 02 · asyncio in depth

Every concept you need to write concurrent Python with `asyncio`, each with a runnable example. Chapter 01 showed *why* async matters for LLM calls. This chapter shows *how* it works.

| # | Concept | Example |
|---|---|---|
| 1 | [Concurrency vs parallelism](#1-concurrency-parallelism-and-where-asyncio-fits) | — |
| 2 | [Coroutines, `async def`, `await`](#2-coroutines-async-def-and-await) | [`01_coroutines.py`](examples/01_coroutines.py) |
| 3 | [Syntax rules](#3-syntax-rules) | — |
| 4 | [The event loop](#4-the-event-loop-and-asynciorun) | [`01_coroutines.py`](examples/01_coroutines.py) |
| 5 | [Chaining coroutines](#5-chaining-coroutines) | [`02_chaining.py`](examples/02_chaining.py) |
| 6 | [Queues](#6-queues-producers-and-consumers) | [`03_queues.py`](examples/03_queues.py) |
| 7 | [Async generators](#7-async-generators-async-for-and-async-comprehensions) | [`04_async_generators.py`](examples/04_async_generators.py) |
| 8 | [`async with`](#8-async-context-managers-async-with) | [`08_async_with_aiohttp.py`](examples/08_async_with_aiohttp.py) |
| 9 | [`create_task`, `gather`, `as_completed`](#9-create_task-gather-and-as_completed) | [`05_tasks_gather_as_completed.py`](examples/05_tasks_gather_as_completed.py) |
| 10 | [`ExceptionGroup` / `except*`](#10-exception-handling-exceptiongroup-and-except) | [`06_exception_groups.py`](examples/06_exception_groups.py) |
| 11 | [`TaskGroup` and `to_thread`](#11-modern-tools-taskgroup-and-to_thread) | [`07_taskgroup_and_to_thread.py`](examples/07_taskgroup_and_to_thread.py) |
| 12 | [When to use asyncio](#12-when-to-use-asyncio-and-the-pitfalls) | — |

---

## 1. Concurrency, parallelism, and where asyncio fits

- **Parallelism** means things literally run at the same time. **Multiprocessing** does this by spreading work across CPU cores, so it's the tool for CPU-bound work.
- **Concurrency** means tasks overlap in time. It doesn't need parallelism. **Threading** gives you concurrency inside one process, with the OS switching between threads.
- **asyncio** gives you concurrency in **one thread and one process**. Tasks *cooperate*: each one explicitly gives up control (`await`) while it waits on I/O, and the event loop runs another task in the meantime.

Think of a chess master playing 30 opponents at once. Instead of finishing one game before starting the next, she makes a move at one board and walks to the next while that opponent thinks. She never plays two moves at the same instant (no parallelism), but no time is wasted waiting (concurrency).

## 2. Coroutines, `async def` and `await`

Calling a function defined with `async def` doesn't run it. It returns a **coroutine object**, which can pause and resume. `await` pauses the current coroutine and hands control back to the event loop until the result is ready.

```python
import asyncio

async def count(name):
    print(f"{name}: one")
    await asyncio.sleep(1)      # pauses this coroutine and lets the others run
    print(f"{name}: two")

async def main():
    await asyncio.gather(count("A"), count("B"), count("C"))

asyncio.run(main())   # 1.00 s: all "one" lines print first, then all "two" lines (a sync version takes 3 s)
```

Never use `time.sleep()` (or any other blocking call) inside async code. It freezes the whole event loop. `asyncio.sleep()` gives control back to the loop.

## 3. Syntax rules

```python
async def ok(x):
    y = await other(x)    # OK: await inside async def
    return y

async def agen():
    yield 1               # OK: this makes it an async generator

def not_ok(x):
    y = await other(x)    # SyntaxError: await outside async def

async def also_bad():
    yield from range(3)   # SyntaxError: yield from isn't allowed in async def
```

You can only `await` an **awaitable**: a coroutine, a `Task`, a `Future`, or any object that implements `__await__()`.

## 4. The event loop and `asyncio.run()`

The event loop watches all coroutines and resumes each one when the thing it's waiting on is ready. A coroutine does nothing until the event loop runs it.

```python
coro = main()        # <coroutine object main at ...>: nothing has run yet
asyncio.run(coro)    # creates the loop, runs it to completion, then closes it
```

- `asyncio.run()` is the standard entry point, called once at the top of your program. It cannot be called when a loop is already running. **Jupyter already runs a loop**, so in a notebook write `await main()` directly instead.
- `asyncio.get_running_loop()` returns the current loop and raises `RuntimeError` if none is running.
- The event loop is swappable. `uvloop` is a faster third-party replacement.
- `python -m asyncio` opens a REPL where you can use `await` at the top level (Python 3.8+).

## 5. Chaining coroutines

When a step needs the previous step's result, await the steps one after another inside a coroutine. Then run several of those chains concurrently.

```python
async def user_with_posts(uid):
    user = await fetch_user(uid)        # step 1
    return await fetch_posts(user)      # step 2 needs step 1's result

async def main():
    return await asyncio.gather(*(user_with_posts(i) for i in range(3)))
```

Each chain runs in order, but the three chains overlap. The total time is roughly the *slowest chain*, not the sum of all the steps. LLM example: retrieve documents, then generate an answer, for many questions at once.

## 6. Queues: producers and consumers

An `asyncio.Queue` connects producers and consumers so neither needs to know about the other. To stop the consumers, the producer puts one **sentinel** value (a "poison pill") per consumer on the queue.

```python
async def producer(q, n_consumers):
    for i in range(5):
        await q.put(i)
    for _ in range(n_consumers):
        await q.put(None)                     # one poison pill per consumer

async def consumer(name, q):
    while (item := await q.get()) is not None:
        print(f"{name} processed {item}")

async def main():
    q = asyncio.Queue()
    await asyncio.gather(producer(q, 3), *(consumer(f"c{i}", q) for i in range(3)))
```

LLM example: one task pages through a document store while N workers embed the chunks.

## 7. Async generators, `async for` and async comprehensions

```python
async def powers_of_two(stop):
    for exp in range(stop):
        yield 2 ** exp
        await asyncio.sleep(0.2)

async def main():
    async for p in powers_of_two(5):
        print(p)                                            # 1 2 4 8 16
    big = [p async for p in powers_of_two(5) if p > 2]      # [4, 8, 16]
```

⚠️ Iteration here is still **sequential**. Concurrency comes only from `gather`, `create_task` or `TaskGroup`. Use `async for` / `async with` only with objects that are actually asynchronous. The token stream of an LLM API is the classic async iterator.

## 8. Async context managers: `async with`

`async with` sets up and cleans up async resources such as HTTP sessions, database connections and streams, even when an error occurs.

```python
import aiohttp

async def check(session, url):
    async with session.get(url) as resp:
        return url, resp.status

async def main():
    urls = ["https://www.python.org", "https://pypi.org"]
    async with aiohttp.ClientSession() as session:          # create once, reuse for all requests
        print(await asyncio.gather(*(check(session, u) for u in urls)))
```

## 9. `create_task`, `gather` and `as_completed`

```python
async def work(name, secs):
    await asyncio.sleep(secs)
    return f"{name} done in {secs}s"

async def main():
    # create_task: starts running in the background right away
    t1 = asyncio.create_task(work("A", 1))
    t2 = asyncio.create_task(work("B", 0.5))
    print(await t1, await t2)        # 1 s total, not 1.5 s

    # gather: results come back in INPUT order
    print(await asyncio.gather(work("A", 1), work("B", 0.5)))   # ['A …', 'B …']

    # as_completed: results come back in FINISH order
    for fut in asyncio.as_completed([work("A", 1), work("B", 0.5)]):
        print(await fut)             # B first, then A
```

- **The most common mistake:** `await work("A", 1); await work("B", 0.5)` runs them one after the other and takes 1.5 seconds. Only `create_task`, `gather` or `TaskGroup` makes them overlap.
- Keep a reference to every task you create, and await it. Tasks still pending when `asyncio.run()` finishes get cancelled.

## 10. Exception handling: `ExceptionGroup` and `except*`

When several tasks fail at once (Python 3.11+), you can bundle their errors into an `ExceptionGroup` and handle each error type separately with `except*`.

```python
async def main():
    results = await asyncio.gather(
        boom(ValueError("v")), boom(TypeError("t")), boom(IndexError("i")),
        return_exceptions=True,        # returns errors as results instead of raising
    )
    errors = [r for r in results if isinstance(r, Exception)]
    if errors:
        raise ExceptionGroup("several failures", errors)

try:
    asyncio.run(main())
except* ValueError as eg:
    print("ValueError:", eg.exceptions)
except* (TypeError, IndexError) as eg:
    print("others:", eg.exceptions)
```

## 11. Modern tools: `TaskGroup` and `to_thread`

```python
# TaskGroup (3.11+): a safer gather. If one task fails, the rest are cancelled
# and the errors raise as an ExceptionGroup.
async def main():
    async with asyncio.TaskGroup() as tg:
        a = tg.create_task(work("A", 2))
        b = tg.create_task(work("B", 1))
    print(a.result(), b.result())     # every task has finished at this point

# asyncio.to_thread (3.9+): runs a blocking sync call without freezing the loop
async def fetch_sync(url):
    return await asyncio.to_thread(requests.get, url)
```

## 12. When to use asyncio, and the pitfalls

| Workload | Best tool |
|---|---|
| Many I/O waits (LLM/HTTP calls, sockets, websockets, chat servers) | **asyncio** |
| CPU-heavy work (number crunching, local model inference) | **multiprocessing** (you can combine it with asyncio) |
| A few blocking calls through sync-only libraries | threading / `asyncio.to_thread` |

- **Never block the event loop.** `time.sleep`, `requests`, ordinary file I/O and sync database drivers inside `async def` freeze every other task.
- **Every library in the chain must be async.** Use `httpx`/`aiohttp` instead of `requests`, async DB drivers (`asyncpg`, …) and the async LLM clients.
- Compared with threading, asyncio has fewer race conditions, because tasks only switch at an `await`. It also scales much further: thousands of tasks are cheap, while threads are not.

---

**Further reading:** [Real Python, *Python's asyncio: A Hands-On Walkthrough*](https://realpython.com/async-io-python/) (the main source for this chapter) · [asyncio docs](https://docs.python.org/3/library/asyncio.html)

**Previous:** [← 01 · Python foundations](../01-python-foundations/README.md)
