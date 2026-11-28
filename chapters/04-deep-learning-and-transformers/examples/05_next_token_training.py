"""Next-token prediction: train a tiny character-level language model, then sample from it.

The model is a "bigram" table: for each character, one score (logit) per possible next
character. It is trained exactly like an LLM, only smaller: self-supervised labels (the next
character), cross-entropy loss, gradient descent. Then it generates text one token at a time.
The last part does the scaling-law arithmetic for real model sizes.

Run: uv run chapters/04-deep-learning-and-transformers/examples/05_next_token_training.py
"""

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]
TEXT = "the cat sat on the mat. the dog ran to the log. a cat and a dog met on a mat. "


def softmax(z: Array) -> Array:
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


if __name__ == "__main__":
    chars = sorted(set(TEXT))  # the vocabulary: every distinct character
    stoi = {c: i for i, c in enumerate(chars)}
    ids = np.array([stoi[c] for c in TEXT])
    # Self-supervision: the text labels itself. The target for each character is the next one.
    inputs, targets = ids[:-1], ids[1:]
    n, v = len(inputs), len(chars)
    print(f"{len(TEXT)} characters of text, vocabulary of {v} characters")
    print(f"guessing uniformly would give a loss of log({v}) = {np.log(v):.2f}\n")

    logits_table = np.zeros((v, v))  # the model's parameters: row = current char, col = next
    for step in range(301):
        probs = softmax(logits_table[inputs])  # (n, v): predicted next-char distributions
        # Cross-entropy: -log(probability given to the right next character), averaged.
        loss = -np.mean(np.log(probs[np.arange(n), targets]))
        if step % 100 == 0:
            print(f"  step {step:3}: loss {loss:.2f}")
        grad = probs.copy()  # for softmax + cross-entropy, d loss / d logits = probs - one_hot
        grad[np.arange(n), targets] -= 1
        d_table = np.zeros_like(logits_table)
        np.add.at(d_table, inputs, grad / n)  # each row collects the gradient of its examples
        logits_table -= 20.0 * d_table  # gradient descent step (learning rate 20)

    # Generation is autoregressive: predict, sample one token, append it, repeat.
    rng = np.random.default_rng(0)
    for temperature in [0.3, 1.0]:
        out = "t"
        for _ in range(45):
            p = softmax(logits_table[stoi[out[-1]]] / temperature)  # low T = sharper choices
            out += chars[rng.choice(v, p=p)]
        print(f"\ntemperature {temperature}: {out!r}")

    print("\nScaling-law arithmetic (rules of thumb, not exact):")
    for params in [1e9, 7e9, 70e9]:
        tokens = 20 * params  # Chinchilla: compute-optimal ~20 training tokens per parameter
        flops = 6 * params * tokens  # training compute ~ 6 x parameters x tokens
        size = f"{params / 1e9:.0f}B parameters"
        print(f"  {size:>15} -> ~{tokens / 1e9:,.0f}B tokens, ~{flops:.1e} FLOPs")
    print("  Many recent models train on far more tokens than this: a smaller model trained")
    print("  longer costs more to train, but less to run for every request afterwards.")
