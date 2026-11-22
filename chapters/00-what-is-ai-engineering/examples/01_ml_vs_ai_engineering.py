"""The same task two ways: traditional ML engineering vs AI engineering.

The task: label short product reviews as "positive" or "negative".
- ML engineering starts from labelled data: choose features, train a model, evaluate it.
- AI engineering starts from a model someone else already trained: write a prompt, evaluate it.
Both are scored by the same `evaluate()` function on the same held-out reviews.

The AI half calls an LLM with the `openai` SDK, configured like chapter 01's llm_client_demo.py.
It defaults to a free local Ollama model:
    ollama pull llama3.2 && ollama serve
Or point it at any OpenAI-compatible API with LLM_BASE_URL, LLM_API_KEY and LLM_MODEL.
Without a reachable server it skips the AI half (the ML half always runs), so the tests pass.

Run: uv run chapters/00-what-is-ai-engineering/examples/01_ml_vs_ai_engineering.py
"""

import math
import os
import re
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

from openai import APIConnectionError, OpenAI

# Where the LLM lives. The defaults point at a local Ollama server (see chapter 01).
BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
API_KEY = os.environ.get("LLM_API_KEY", "ollama")  # Ollama ignores it, the SDK requires one
MODEL = os.environ.get("LLM_MODEL", "llama3.2")

Example = tuple[str, str]  # (review text, label)

# Labelled data: the starting point of ML engineering. A real project needs thousands of these,
# collected and labelled by people before any model exists.
TRAIN: list[Example] = [
    ("great sound and the battery lasts all day", "positive"),
    ("love it, works perfectly", "positive"),
    ("excellent quality for the price", "positive"),
    ("fast delivery and friendly support", "positive"),
    ("comfortable and easy to use", "positive"),
    ("works great, I recommend it", "positive"),
    ("perfect fit and a lovely colour", "positive"),
    ("good value, would buy again", "positive"),
    ("terrible sound, broke after a week", "negative"),
    ("waste of money, it does not work", "negative"),
    ("poor quality and slow delivery", "negative"),
    ("the battery died, very disappointed", "negative"),
    ("cheap plastic that broke on day one", "negative"),
    ("rude support, never again", "negative"),
    ("uncomfortable and hard to use", "negative"),
    ("bad value, I returned it", "negative"),
]

# Held-out reviews that neither approach sees in advance. Some are hard on purpose: words that
# never appear in TRAIN, and negations ("not bad") that flip the meaning of a word.
TEST: list[Example] = [
    ("love the sound, great battery", "positive"),
    ("broke after two days, a waste of money", "negative"),
    ("exceeded my expectations", "positive"),
    ("arrived scratched and the seller ignored me", "negative"),
    ("not bad at all, works fine", "positive"),
    ("I would not recommend it", "negative"),
    ("solid build, my kids adore it", "positive"),
    ("stopped charging within a month", "negative"),
]


# ----------------------------------------------------------------------------------------------
# 1. ML engineering: features + a model trained from scratch on our own labelled data
# ----------------------------------------------------------------------------------------------
def tokenize(text: str) -> list[str]:
    """Feature engineering, in its simplest form: lowercase words, order ignored (bag of words)."""
    return re.findall(r"[a-z']+", text.lower())


@dataclass
class NaiveBayes:
    """A classic text classifier: it learns how often each word appears with each label."""

    label_counts: Counter[str] = field(default_factory=Counter)  # label -> number of reviews
    word_counts: dict[str, Counter[str]] = field(default_factory=dict)  # label -> word counts
    vocabulary: set[str] = field(default_factory=set)  # every word seen in training

    def train(self, examples: list[Example]) -> None:
        # "Training" here is just counting. Bigger models learn millions of numbers instead,
        # but the workflow is the same: labelled data in, model parameters out.
        for text, label in examples:
            words = tokenize(text)
            self.label_counts[label] += 1
            self.word_counts.setdefault(label, Counter()).update(words)
            self.vocabulary.update(words)

    def predict(self, text: str) -> str:
        total_reviews = sum(self.label_counts.values())
        scores: dict[str, float] = {}
        for label, n_reviews in self.label_counts.items():
            counts = self.word_counts[label]
            n_words = sum(counts.values())
            # Score = log P(label) + sum of log P(word | label). Adding logs instead of
            # multiplying probabilities avoids numbers too small for a float.
            score = math.log(n_reviews / total_reviews)
            for word in tokenize(text):
                if word not in self.vocabulary:
                    continue  # a word never seen in training tells the model nothing
                # +1 ("Laplace smoothing"): a word seen only with the other label must not get
                # probability 0, or it would veto every other word in the review.
                score += math.log((counts[word] + 1) / (n_words + len(self.vocabulary)))
            scores[label] = score
        # The label with the highest score wins. On a tie, the first label in TRAIN wins.
        return max(scores, key=scores.__getitem__)


# ----------------------------------------------------------------------------------------------
# 2. AI engineering: a pre-trained foundation model + a prompt
# ----------------------------------------------------------------------------------------------
# The prompt is the "program": it describes the task and the exact answer format we want.
PROMPT = (
    "Classify the sentiment of this product review.\n"
    "Answer with exactly one word: positive or negative.\n\n"
    "Review: {review}"
)


def classify_with_llm(client: OpenAI, review: str) -> str:
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT.format(review=review)}],
        max_tokens=5,  # one word is enough: a cap keeps cost and latency low
        temperature=0,  # pick the most likely answer every time, for repeatable results
    )
    answer = (completion.choices[0].message.content or "").strip().lower()
    # The output is free text, not a label: the model may say "Positive." or a whole sentence.
    # Map it onto our labels, and count anything else as wrong. Deciding what counts as a
    # correct answer is part of the evaluation work.
    for label in ("positive", "negative"):
        if answer.startswith(label):
            return label
    return f"unparsable: {answer!r}"


# ----------------------------------------------------------------------------------------------
# Evaluation: the same yardstick for both approaches
# ----------------------------------------------------------------------------------------------
def evaluate(name: str, classify: Callable[[str], str], test_set: list[Example]) -> int:
    """Run `classify` on every test review, print each result, return how many were right."""
    print(f"\n{name}")
    correct = 0
    for text, expected in test_set:
        predicted = classify(text)
        ok = predicted == expected
        correct += ok  # True counts as 1, False as 0
        print(f"  {'✓' if ok else '✗'} {text!r:48} -> {predicted} (expected {expected})")
    print(f"  accuracy: {correct}/{len(test_set)}")
    return correct


if __name__ == "__main__":
    print(f"Task: label product reviews as positive or negative ({len(TEST)} test reviews)")

    print("\n=== ML engineering: start from labelled data, train a model ===")
    start = time.perf_counter()
    model = NaiveBayes()
    model.train(TRAIN)
    elapsed_ms = (time.perf_counter() - start) * 1000
    print(f"trained on {len(TRAIN)} labelled reviews in {elapsed_ms:.2f} ms;")
    print(f"the model knows {len(model.vocabulary)} words and nothing else about language")
    ml_correct = evaluate("Naive Bayes, trained here from scratch:", model.predict, TEST)

    print("\n=== AI engineering: start from a pre-trained model, write a prompt ===")
    # Health check: a cheap request with a short timeout and no retries, so a missing server
    # fails fast. No server is an expected situation, not a bug: skip this half and exit 0.
    try:
        OpenAI(base_url=BASE_URL, api_key=API_KEY, max_retries=0, timeout=5).models.list()
    except APIConnectionError:
        print(f"No LLM server reachable at {BASE_URL}: skipping the AI engineering half.")
        print("Run `ollama serve` (see chapter 01) to compare both approaches.")
        raise SystemExit(0) from None

    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    print(f"model: {MODEL} at {BASE_URL}, trained by someone else on far more text than ours")
    ai_correct = evaluate(
        f"{MODEL}, zero-shot prompt (0 labelled examples):",
        lambda text: classify_with_llm(client, text),
        TEST,
    )

    print("\n=== What changed ===")
    rows = [
        ("starting point", "labelled data", "a pre-trained model"),
        ("labelled examples used", str(len(TRAIN)), "0"),
        ("what you wrote", "features + training", "a prompt"),
        ("output", "always a label", "free text to check"),
        ("correct on the test set", f"{ml_correct}/{len(TEST)}", f"{ai_correct}/{len(TEST)}"),
    ]
    print(f"  {'':26}{'ML engineering':22}AI engineering")
    for what, ml, ai in rows:
        print(f"  {what:26}{ml:22}{ai}")
