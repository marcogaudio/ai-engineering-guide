# NN · Chapter title

One or two sentences: what this chapter covers and why it matters for building LLM applications.

| Topic | Why it matters | Example |
|---|---|---|
| [Concept A](#1-concept-a) | … | [`concept_a.py`](examples/concept_a.py) |
| [Concept B](#2-concept-b) | … | [`concept_b.py`](examples/concept_b.py) |

---

## 1. Concept A

**What it is:** one short paragraph in plain words.

```python
# the smallest snippet that shows the idea
```

**In LLM work:** where you meet it and what goes wrong without it.

## 2. Concept B

…

---

**Further reading:** [source 1](https://…) · [source 2](https://…)

**Previous:** [← NN-1 · …](../NN-1-topic/README.md) · **Next:** [NN+1 · … →](../NN+1-topic/README.md)

<!--
Checklist for a new chapter:
- [ ] examples/*.py run with `uv run <file>` and exit 0 (needing an API key → skip gracefully when it's missing)
- [ ] network-only examples contain the line `requires: network` in their docstring
- [ ] chapter added to the table in the root README.md
- [ ] `uv run pytest && uv run ruff check . && uv run mypy chapters` all pass
-->
