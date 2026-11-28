# 00 · What is AI engineering?

AI engineering is building applications on top of foundation models that already exist, rather than training models from scratch. This guide follows that framing, from Chip Huyen's *AI Engineering* (chapter 1). This chapter explains what changes when the model comes pre-built, and what that means for the work you do.

| Topic | Why it matters | Runnable example |
|---|---|---|
| [What AI engineering is](#1-what-ai-engineering-is) | The definition the rest of this guide builds on | — |
| [AI engineering vs ML engineering](#2-ai-engineering-vs-ml-engineering) | A different starting point, and different core work | [`01_ml_vs_ai_engineering.py`](examples/01_ml_vs_ai_engineering.py) (the ML half runs anywhere; the AI half uses a free local model via Ollama, or any OpenAI-compatible API) |
| [AI engineering vs full-stack engineering](#3-ai-engineering-vs-full-stack-engineering) | Product first; the hard part is making it reliable | — |
| [The three layers of the AI stack](#4-the-three-layers-of-the-ai-stack) | Know which layer your work lives in | — |
| [Why this matters for your career](#5-why-this-matters-for-your-career) | What sets AI engineers apart, and how this guide is organized | — |

---

## 1. What AI engineering is

**What it is:** building products on top of a **foundation model**. That's a large model someone else has already trained, which you usually reach through an API. You don't collect a dataset or train anything to get started: you describe the task to the model, and build the product around its answers.

How we got here, in three steps:
- A **language model** predicts the next piece of text (a *token*) from the text before it.
- A **large language model (LLM)** is the same idea at a much bigger scale. What made that scale possible is **self-supervision**: text supplies its own labels, because each next word is the "right answer" for the words before it. So models can learn from huge amounts of text that nobody had to label by hand.
- A **foundation model** is a model general enough to be adapted to many different tasks, often across several kinds of data (text, images, audio).

Huyen's point is that foundation models are what turned AI engineering into a discipline of its own. Because they're offered as a service, the barrier to entry is low: a first working AI feature can be a prompt and one API call. Here it is with the `openai` SDK and a local model, as set up in [Chapter 01](../01-python-foundations/README.md):

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
prompt = "Is this review positive or negative? One word.\n\nArrived scratched and the seller ignored me."
reply = client.chat.completions.create(model="llama3.2", messages=[{"role": "user", "content": prompt}])
print(reply.choices[0].message.content)   # e.g. "Negative." (free text, not a label)
```

**In practice:** what people build on foundation models falls into a few families, which Huyen catalogues:
- coding;
- image and video production;
- writing;
- education;
- conversational bots;
- information aggregation (summaries, "talk to your docs");
- data organization;
- workflow automation.

## 2. AI engineering vs ML engineering

**What changes:**

| | Traditional ML engineering | AI engineering |
|---|---|---|
| Starting point | Collect and label data, then train your own model | Pick an existing foundation model |
| How you shape it | Feature engineering, often on tabular data | Prompts, and the context you put in front of the model |
| Where the effort goes | Training the model | Adapting the model, and evaluating what it produces |
| Time to something that works | Weeks to a first model | Minutes to a first prototype |

Three consequences follow, and the rest of this guide is built around them:
- **You adapt the model instead of training it.** The first tools are prompt engineering and context construction: giving the model the right information at the right moment, for example with RAG. Finetuning changes the model's weights, and comes later, only when prompts and context aren't enough.
- **The models are big.** Every call takes time and costs money, so latency and token cost become everyday engineering concerns (chapters 05 and 11).
- **The outputs are open-ended.** A classic classifier returns one of a few labels. An LLM returns free text, and often there's no single right answer. That makes quality harder to check, so evaluation becomes a core skill rather than a final step (chapter 09).

ML engineering doesn't go away: AI engineering grew out of it, and its fundamentals still apply. Those are good data, clear metrics and held-out test sets.

[`01_ml_vs_ai_engineering.py`](examples/01_ml_vs_ai_engineering.py) does the same task both ways: labelling eight product reviews it has never seen as positive or negative.

**The ML engineering half** trains a tiny Naive Bayes classifier from scratch on 16 labelled reviews.
- It knows 62 words and nothing else about language. All three of its mistakes were complaints it called "positive", such as "stopped charging within a month".
- Even one of its five correct answers was luck: none of the words in "exceeded my expectations" were in its training data, so it fell back on a tie-break.

**The AI engineering half** sends one prompt per review to `llama3.2`, with zero labelled examples.
- In our run it got 7 of 8 right, against 5 of 8 for the classifier.
- It missed "not bad at all, works fine". A pre-trained model isn't automatically right, which is why both halves go through the same `evaluate()` function.
- Its answers are free text that has to be mapped back to a label, and the script counts anything else as wrong.

Your numbers may differ with another model. To be fair to ML engineering: with thousands of labelled reviews, a trained classifier would do much better, at a fraction of the cost per prediction. The difference is that AI engineering lets you *start* without that data.

## 3. AI engineering vs full-stack engineering

**What changes:** when the model is an API call, building an AI feature looks a lot like ordinary product engineering, and full-stack engineers can ship one quickly. The order of the work flips:

```text
ML engineering:   data → model → product
AI engineering:   product → data → model
```

You build a working demo on an existing model first and put it in front of users. Only once the product shows promise do you invest in data (evaluation sets, examples, finetuning data) or in a different model.

**What carries over** from full-stack work:
- The interface matters even more: chat UIs, streaming tokens, ways for users to give feedback.
- So does everything around it: APIs, authentication, deployment.

**What's new:**
- The core component is non-deterministic, and its output is open-ended.
- The same input can give different answers, and a confident answer can be wrong.
- A prompt that works in a demo can fail on the next hundred real inputs.

Getting from a good demo to a reliable product is where most of the engineering effort goes. That means evaluation, grounding the model in real data, guardrails, and cost and latency budgets.

## 4. The three layers of the AI stack

![Three stacked layers. Top: application development (prompts, context, evaluation, the product), where most AI engineers work. Middle: model development (training, finetuning, dataset engineering, inference optimization). Bottom: infrastructure (serving, compute, monitoring). With a hosted model API, the provider runs most of the bottom two layers.](images/ai-stack.svg)

| Layer | What lives there | In this guide |
|---|---|---|
| **Application development** | Prompts, context, evaluation, the product itself | The core of the guide: chapters 05–10 |
| **Model development** | Training, finetuning, dataset engineering, inference optimization | Mostly out of scope: you build on models that others trained and optimized. Chapter 04 explains how they work inside |
| **Infrastructure** | Serving, compute, monitoring | Running and monitoring *your* application: chapters 11–12 |

- **Most AI engineers work in the top layer**, and so does this guide.
- **With a hosted model API** (Anthropic, OpenAI, Google, …), the provider takes care of developing, serving and scaling the model itself. You still own the infrastructure of your own application: running it, monitoring it, and keeping it within budget.
- **Running a model locally with Ollama**, as the examples here do by default, gives you a small taste of the infrastructure layer: you are serving a model yourself.

## 5. Why this matters for your career

When anyone can call the same models, being able to train one is no longer what sets an AI engineer apart. What does is being able to **evaluate** an AI system, **ground** it in the right data, and **ship** it so it works reliably for real users. That's why this guide spends whole chapters on those skills:

| Skill | Chapters |
|---|---|
| Talk to models well | 05 Calling LLM APIs · 06 Structured outputs & tool use |
| Ground answers in your own data | 07 Embeddings & vector search · 08 RAG |
| Measure quality | 09 Evals |
| Let models act, safely | 10 Agents (tools, memory, guardrails) |
| Ship it and keep it running | 11 Observability & cost · 12 Deployment |

Chapters 01–03 come first because all of that is written in Python and talks to models over HTTP: they give you the tools. [Chapter 04](../04-deep-learning-and-transformers/README.md) then opens up the model itself: neural networks, Transformers, tokens and how LLMs are trained.

---

**How they fit together:** AI engineering starts from a model that already exists. You adapt it with prompts and context instead of training it. You evaluate what it produces, because its output is open-ended. And you build the product around it, mostly in the application layer of the stack. The rest of this guide covers each of those steps.

**Further reading:** [Chip Huyen, *AI Engineering*, ch. 1: Introduction to Building AI Applications with Foundation Models](https://www.oreilly.com/library/view/ai-engineering/9781098166298/) (the main source for this chapter) · [the book's companion repo](https://github.com/chiphuyen/aie-book) (table of contents and resources) · [its chapter summaries](https://github.com/chiphuyen/aie-book/blob/main/chapter-summaries.md)

**Next:** [01 · Python foundations →](../01-python-foundations/README.md)
