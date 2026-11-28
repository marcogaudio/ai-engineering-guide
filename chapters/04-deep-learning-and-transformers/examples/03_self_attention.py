"""Self-attention and one Transformer block, in numpy, on 4 toy tokens.

The weights are random: in a real model they are learned during training. What matters here is
the data flow and the shapes, which are exactly those of a GPT-style (decoder-only) model.

Run: uv run chapters/04-deep-learning-and-transformers/examples/03_self_attention.py
"""

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]
rng = np.random.default_rng(0)
TOKENS = ["the", "cat", "sat", "down"]
D = 8  # embedding size: numbers per token (real LLMs use thousands)


def weights(rows: int, cols: int) -> Array:
    return rng.normal(0, 1 / np.sqrt(rows), (rows, cols))  # small random matrix


def positional_encoding(n: int, d: int) -> Array:
    """Sinusoidal positions (original Transformer): a different wave pattern per position."""
    pos, i = np.arange(n)[:, None], np.arange(d)[None, :]
    angle = pos / 10000 ** (2 * (i // 2) / d)
    return np.where(i % 2 == 0, np.sin(angle), np.cos(angle))


def softmax(z: Array) -> Array:
    e = np.exp(z - z.max(axis=-1, keepdims=True))  # minus the max: same result, no overflow
    return e / e.sum(axis=-1, keepdims=True)


def attention(x: Array, wq: Array, wk: Array, wv: Array) -> tuple[Array, Array]:
    """Causal self-attention: each token mixes the values of itself and the tokens before it."""
    q, k, v = x @ wq, x @ wk, x @ wv  # query: what I look for; key: what I hold; value: what I give
    scores = q @ k.T / np.sqrt(k.shape[-1])  # how well each query matches each key, scaled
    future = np.triu(np.ones(scores.shape, dtype=bool), k=1)
    scores = np.where(future, -np.inf, scores)  # causal mask: no peeking at later tokens
    attn = softmax(scores)  # each row: attention weights that sum to 1
    return attn @ v, attn  # each output row is a weighted average of the value rows


def multi_head(x: Array, n_heads: int = 2) -> Array:
    """Several smaller attention heads side by side, concatenated, then mixed by one matrix."""
    d_head = D // n_heads
    outs = [
        attention(x, weights(D, d_head), weights(D, d_head), weights(D, d_head))[0]
        for _ in range(n_heads)
    ]
    return np.concatenate(outs, axis=-1) @ weights(D, D)


def layer_norm(x: Array) -> Array:
    """Rescale each token's vector to mean 0 and variance 1: keeps numbers stable in deep stacks."""
    return (x - x.mean(axis=-1, keepdims=True)) / np.sqrt(x.var(axis=-1, keepdims=True) + 1e-5)


def feed_forward(x: Array) -> Array:
    """A small 2-layer network applied to every token separately, 4x wider in the middle."""
    return np.maximum(0, x @ weights(D, 4 * D)) @ weights(4 * D, D)  # ReLU in between


def transformer_block(x: Array) -> Array:
    x = x + multi_head(layer_norm(x))  # attention: tokens exchange information (+ residual)
    x = x + feed_forward(layer_norm(x))  # feed-forward: each token is processed on its own
    return x


if __name__ == "__main__":
    embeddings = rng.normal(0, 1, (len(TOKENS), D))  # one vector per token (learned, in reality)
    x = embeddings + positional_encoding(len(TOKENS), D)  # add "where am I" information
    print("input to the first block:", x.shape, "= (tokens, numbers per token)")

    _, attn = attention(x, weights(D, D), weights(D, D), weights(D, D))
    print("\nattention weights, one head (row = token looking, column = token looked at):")
    print("        " + "".join(f"{t:>7}" for t in TOKENS))
    for token, row in zip(TOKENS, attn, strict=True):
        print(f"  {token:>5} " + "".join(f"{w:7.2f}" for w in row))
    print("  every row sums to 1; the 0.00s above the diagonal are the causal mask")

    for n in range(1, 3):  # real LLMs stack dozens of blocks; each keeps the shape unchanged
        x = transformer_block(x)
        print(f"\nafter block {n}:", x.shape)

    logits = layer_norm(x) @ weights(D, len(TOKENS))  # final norm, then one score per vocab entry
    probs = softmax(logits[-1])  # the last position predicts the next token
    print(
        "\nnext-token probabilities after 'the cat sat down':",
        {t: round(float(p), 2) for t, p in zip(TOKENS, probs, strict=True)},
    )
    print("(the weights are random, so these are meaningless until the model is trained)")
