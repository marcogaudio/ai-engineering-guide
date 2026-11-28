"""Byte-pair encoding (BPE) from scratch, then token ids -> embedding vectors.

BPE starts from single characters and repeatedly merges the most frequent adjacent pair into a
new token. Frequent words end up as one token; rare words are split into known pieces.
The toy corpus is the classic one from Sennrich et al. (2016), who brought BPE to NLP.

Run: uv run chapters/04-deep-learning-and-transformers/examples/04_bpe_and_embeddings.py
"""

from collections import Counter
from itertools import pairwise

import numpy as np

CORPUS = {"low": 5, "lower": 2, "newest": 6, "widest": 3}  # word -> how often it appears
END = "_"  # marks the end of a word, so "est_" (a word ending) differs from "est" inside a word


def merge_pair(symbols: list[str], pair: tuple[str, str]) -> list[str]:
    """Replace every adjacent occurrence of `pair` with one merged symbol."""
    out, i = [], 0
    while i < len(symbols):
        if symbols[i : i + 2] == list(pair):
            out.append(pair[0] + pair[1])
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return out


def train_bpe(corpus: dict[str, int], n_merges: int) -> list[tuple[str, str]]:
    words = {word: list(word) + [END] for word in corpus}  # start: one symbol per character
    merges: list[tuple[str, str]] = []
    for _ in range(n_merges):
        pairs: Counter[tuple[str, str]] = Counter()
        for word, symbols in words.items():  # count adjacent pairs, weighted by word frequency
            for pair in pairwise(symbols):  # ("l","o"), ("o","w"), ("w","_") for "low"
                pairs[pair] += corpus[word]
        best = max(pairs, key=pairs.__getitem__)  # the most frequent pair becomes a token
        merges.append(best)
        words = {word: merge_pair(symbols, best) for word, symbols in words.items()}
    return merges


def tokenize(word: str, merges: list[tuple[str, str]]) -> list[str]:
    symbols = list(word) + [END]
    for pair in merges:  # replay the merges in the order they were learned
        symbols = merge_pair(symbols, pair)
    return symbols


if __name__ == "__main__":
    merges = train_bpe(CORPUS, n_merges=10)
    print("learned merges, in order:", ", ".join(f"{a}+{b}" for a, b in merges))

    # The vocabulary: every single character, plus one new token per merge. Each gets an id.
    chars = sorted({c for word in CORPUS for c in word} | {END})
    vocab = chars + [a + b for a, b in merges]
    token_id = {token: i for i, token in enumerate(vocab)}
    print(f"vocabulary: {len(vocab)} tokens = {len(chars)} characters + {len(merges)} merges\n")

    for word in ["newest", "lowest", "wider", "zero"]:
        tokens = tokenize(word, merges)
        ids = [token_id.get(t, -1) for t in tokens]  # -1: not in the vocabulary
        print(f"  {word:7} ({len(word)} chars) -> {len(tokens)} token(s) {tokens} ids {ids}")
    print("  'z' never appears in the corpus, so it has no token (id -1). Real tokenizers")
    print("  start from the 256 possible bytes instead of characters, so nothing is unknown.")

    # Embeddings: a matrix with one row of D numbers per token id. Text -> ids -> rows.
    rng = np.random.default_rng(0)
    embedding_matrix = rng.normal(0, 1, (len(vocab), 8))  # learned in training; random here
    ids = [token_id[t] for t in tokenize("lowest", merges)]
    vectors = embedding_matrix[ids]  # one lookup: no maths, just picking rows
    print(f"\n'lowest' -> ids {ids} -> vectors of shape {vectors.shape}")
    print("  (tokens, numbers per token): this matrix is what the first Transformer block reads")
    print("first vector:", vectors[0].round(2))
