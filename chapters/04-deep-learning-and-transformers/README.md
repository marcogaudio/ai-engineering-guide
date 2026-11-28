# 04 · Deep learning and Transformer foundations

You won't train an LLM in this guide (see [Chapter 00](../00-what-is-ai-engineering/README.md)), but knowing what's inside one explains most of what you'll meet when you use it: why you pay per token, why there's a context limit, what temperature does, and why a model can sound confident and still be wrong. Five ideas, each with plain definitions, a short snippet and a small numpy script.

| Topic | Why it matters for LLM work | Runnable example |
|---|---|---|
| [Neural network fundamentals](#1-neural-network-fundamentals) | An LLM is a (very large) neural network | [`01_neural_network.py`](examples/01_neural_network.py) |
| [Why sequence modeling is hard](#2-why-sequence-modeling-is-hard) | The problem attention was invented to solve | [`02_rnn_limits.py`](examples/02_rnn_limits.py) |
| [The Transformer architecture](#3-the-transformer-architecture) | The architecture inside every modern LLM | [`03_self_attention.py`](examples/03_self_attention.py) |
| [Tokenization and embeddings](#4-tokenization-and-embeddings) | Prices, limits and quirks are all counted in tokens | [`04_bpe_and_embeddings.py`](examples/04_bpe_and_embeddings.py) |
| [How LLMs are trained](#5-how-llms-are-trained) | Explains temperature, hallucinations and model sizes | [`05_next_token_training.py`](examples/05_next_token_training.py) |

All five examples run offline in about a second, with numpy and no GPU.

---

## 1. Neural network fundamentals

**Definitions:**
- **Neuron:** multiplies each of its inputs by a *weight*, adds them up plus a *bias*, and passes the result through an activation function.
- **Layer:** a group of neurons that all read the same inputs. A network stacks layers: inputs → hidden layers → output.
- **Activation function:** a simple non-linear function, such as ReLU (`max(0, z)`) or sigmoid (squashes any number into 0..1). Without it, a stack of layers collapses into one linear function.
- **Loss:** one number that says how wrong the predictions are, for example the mean squared error. Training means making it small.
- **Gradient descent:** the gradient says, for each weight, which direction makes the loss grow. Move every weight a small step the other way. The size of the step is the *learning rate*.
- **Backpropagation:** the chain rule applied backwards, from the loss to every weight, so one backward pass gives all the gradients.

![A network with two inputs, a hidden layer of four neurons and one output, followed by the loss. The forward pass goes from inputs to prediction to loss; backpropagation goes back and gives the gradient of the loss for every weight; gradient descent then updates each weight as w minus learning rate times gradient.](images/neural-network.svg)

```python
h = relu(x @ W1 + b1)                # forward: hidden layer
y_hat = h @ W2 + b2                  # forward: output
loss = ((y_hat - y) ** 2).mean()     # how wrong?
# backward: the chain rule gives dW1 and dW2, then gradient descent:
W1 -= learning_rate * dW1
W2 -= learning_rate * dW2
```

[`01_neural_network.py`](examples/01_neural_network.py) trains a 2 → 4 → 1 network (17 parameters) on XOR. The loss falls from 0.34 to 0.0004 in 4,000 steps, and the predictions end at 0.01, 0.98, 0.98 and 0.03 for the right answers 0, 1, 1, 0. The same loop with no hidden layer gets stuck at 0.5 for every input: one straight line can't separate XOR.

**In LLM work:** an LLM is the same recipe at an enormous scale. Its *parameters* are all its weights and biases: a "7B" model has about 7 billion of them, and they're what you download when you run a model locally.

## 2. Why sequence modeling is hard

Text is a sequence: the meaning of a word depends on the words before it, sometimes far before.

**Definitions:**
- **Sequence model:** a model that reads an ordered list of items (tokens) and uses their order.
- **RNN (recurrent neural network):** reads one token at a time and keeps a *hidden state*, a fixed-size vector updated at every step: `h = tanh(W·h + U·x)`.
- **Vanishing / exploding gradients:** in training, the signal from the end of a sequence back to its start is a product of one factor per step. Factors below 1 shrink it towards 0; factors above 1 blow it up.
- **LSTM / GRU:** RNN variants with *gates* that decide what to keep and what to forget. They cope better with long sequences, but still read one step at a time.
- **Attention:** let every position look directly at every other position, and learn how much each one matters.

RNNs have three problems:
1. **They're sequential.** Step 50 has to wait for step 49, so training can't process the positions in parallel on a GPU.
2. **There's a bottleneck.** Everything read so far has to fit in one fixed-size hidden state.
3. **Their memory fades.** Information from early tokens gets weaker with every step (vanishing gradients).

```python
h = np.zeros(d)
for x in tokens:                     # one step at a time: no parallelism
    h = np.tanh(W @ h + U @ x)       # everything so far squeezed into h
```

[`02_rnn_limits.py`](examples/02_rnn_limits.py) measures how much the last hidden state still depends on the *first* input. For a small tanh RNN it's 0.98 after 1 step, 0.00096 after 10 steps, and 6e-27 after 50. A linear recurrence with a weight of 1.5 explodes instead, to 4e8 after 50 steps. With attention, the last token reaches the first one in a single step, whatever the length.

**In LLM work:** attention is what made long context windows practical, because any token can use any earlier token directly. The price is that every token is compared with every other one (an n × n table of scores), which is one reason long prompts are slower to process.

## 3. The Transformer architecture

The Transformer ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)) drops recurrence entirely and builds everything on attention.

**Definitions:**
- **Query, key, value:** three vectors computed from each token. The query is what the token is looking for, the key is what it contains, and the value is what it hands over when chosen.
- **Self-attention:** each token scores every token (query · key), turns the scores into weights with softmax, and outputs the weighted average of the values: `softmax(Q·Kᵀ / √d) · V`. Dividing by √d keeps the scores in a sensible range.
- **Causal mask:** in a model that generates text, each token may attend only to itself and earlier tokens, never to future ones.
- **Multi-head attention:** several attention operations side by side, each with its own query, key and value weights, so each head can track a different kind of relationship. Their outputs are concatenated.
- **Positional encoding:** attention by itself ignores order ("cat sat" looks like "sat cat"), so position information is added to each token's vector. The original paper used sine waves; modern LLMs use learned or rotary (RoPE) positions.
- **Feed-forward block:** a small two-layer network applied to each token separately, usually 4× wider in the middle. Attention moves information *between* tokens; the feed-forward block processes each token *on its own*.

Two more pieces hold the stack together. **Residual connections** add each sub-layer's output to its input (`x + f(x)`), so information and gradients flow through deep stacks. **Layer normalization** rescales each token's vector to keep the numbers stable.

```python
q, k, v = x @ Wq, x @ Wk, x @ Wv
scores = q @ k.T / np.sqrt(d)
scores[np.triu_indices(n, k=1)] = -np.inf    # causal mask: no looking ahead
weights = softmax(scores)                    # each row sums to 1
out = weights @ v                            # each token: a weighted mix of the values
```

![A decoder-only Transformer, top to bottom: tokens, then embeddings plus positions, then a Transformer block repeated N times. The block is LayerNorm, masked multi-head self-attention and a residual connection, then LayerNorm, a feed-forward network applied to each token, and a second residual connection. After the blocks: a final LayerNorm, then linear plus softmax, giving the next-token probabilities.](images/transformer-block.svg)

Modern LLMs are **decoder-only** (GPT-style): embeddings, then the same block repeated N times (dozens of times in large models), then a final layer that turns each vector into a probability for every token in the vocabulary. The diagram shows the *pre-norm* layout that modern LLMs use; the original paper normalized after each sub-layer instead.

[`03_self_attention.py`](examples/03_self_attention.py) runs this on four toy tokens with 8 numbers each, using random weights. One head's attention weights show the causal mask at work: "the" can only look at itself (1.00), "cat" splits 0.94 / 0.06 between "the" and itself, every row sums to 1, and everything above the diagonal is 0. Two pre-norm blocks follow, and the shape stays `(4, 8)` after each one. That's why blocks can be stacked.

**In LLM work:** the context window is the number of positions the model was built and trained to handle. Every token of your prompt passes through every block, so longer prompts and bigger models both mean more computation.

## 4. Tokenization and embeddings

Models don't read characters or words. They read **tokens**, and each token becomes a vector of numbers.

**Definitions:**
- **Token:** a piece of text: a whole common word, part of a rarer word, a punctuation mark or a space. In English a token is roughly 4 characters on average; other languages and code usually need more tokens for the same text.
- **Tokenizer and vocabulary:** the tokenizer splits text into tokens from a fixed **vocabulary**, where each token has an integer id. LLM vocabularies usually hold tens of thousands to a few hundred thousand tokens.
- **BPE (byte-pair encoding):** builds the vocabulary from data. Start from single characters, find the most frequent adjacent pair, merge it into a new token, and repeat. Frequent words end up as one token; rare words are split into known pieces. Modern tokenizers run BPE on *bytes*, so any text can be encoded and nothing is "unknown".
- **Context window (token limit):** the maximum number of tokens a model handles in one request, input and output together.
- **Embedding:** the vector of numbers for one token: one row of a learned matrix of shape `vocabulary size × d`. Turning ids into vectors is just a lookup.

```python
ids = [token_id[t] for t in tokens]     # text -> tokens -> ids
vectors = embedding_matrix[ids]         # ids -> an (n_tokens, d) matrix: the model's input
```

[`04_bpe_and_embeddings.py`](examples/04_bpe_and_embeddings.py) trains BPE on the classic toy corpus from [Sennrich et al.](https://arxiv.org/abs/1508.07909) (low, lower, newest, widest).
- Its first merges are `e+s`, `es+t` and `est+_`, so the word ending "est" becomes a single token.
- After 10 merges the vocabulary has 21 tokens: "newest" is one token, the unseen word "lowest" splits into `low` + `est_`, and "zero" contains a "z" the corpus never had, so it gets no id.
- The ids of "lowest" then pick 2 rows of an embedding matrix: a `(2, 8)` array, ready for the first Transformer block.

**In LLM work:**
- Prices, rate limits and context windows are all counted in tokens, not characters or words (chapter 05).
- Models are weak at character-level tasks, such as counting the letters in a word, because they see tokens, not letters.
- Every provider has its own tokenizer, so the same text can be a different number of tokens on different models.
- These per-token embeddings live inside the model. The embeddings of whole texts, used for search, come in chapter 07.

## 5. How LLMs are trained

**Definitions:**
- **Next-token prediction:** given the tokens so far, output a probability for every possible next token. It's the only task a base LLM is trained on.
- **Self-supervision:** the text provides its own labels: the right answer at each position is simply the next token (see [Chapter 00](../00-what-is-ai-engineering/README.md#1-what-ai-engineering-is)).
- **Cross-entropy loss:** `−log(probability the model gave to the actual next token)`, averaged over the text. Guessing uniformly among V tokens scores `log V`; a perfect model scores 0.
- **Pretraining:** next-token prediction over trillions of tokens of web pages, books and code, where almost all the compute goes. **Post-training** (instruction tuning, then preference tuning such as RLHF) then turns this text completer into a helpful assistant.
- **Autoregressive generation:** predict a distribution, sample one token, append it, repeat. **Temperature** divides the scores before the softmax: low values give safer, more repetitive text, high values more varied text.
- **Scaling laws:** the loss falls smoothly and predictably as parameters, data and compute grow ([Kaplan et al., 2020](https://arxiv.org/abs/2001.08361)). A rule of thumb for training compute is `6 × parameters × tokens` FLOPs.

```python
probs = softmax(logits)                               # one probability per vocabulary token
loss = -np.log(probs[np.arange(n), targets]).mean()   # cross-entropy on the real next tokens
next_id = rng.choice(vocab_size, p=softmax(logits[-1] / temperature))   # sample the next token
```

[`05_next_token_training.py`](examples/05_next_token_training.py) trains a character-level model, a 15 × 15 table of scores, on one line of text.
- The loss starts at log(15) = 2.71, the score of uniform guessing, and falls to 1.07.
- Sampling at temperature 0.3 gives `the athe me at t the a the the t athe the at c`. At 1.0 it gives the more varied `the athet d log lo mat t. on t. ondon me rat r`.
- It's the same mechanism as an LLM, with a table in place of a Transformer.

The script also does the **Chinchilla** arithmetic ([Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556)). For a fixed compute budget, the best loss comes from scaling model size and training tokens together, at about 20 tokens per parameter. That's ~140B tokens and ~5.9e21 FLOPs for a 7B model. Many recent models are trained on far more tokens than that: a smaller model trained for longer costs more to train, but less to run on every request afterwards.

**In LLM work:**
- The model always produces a *plausible continuation*, not a checked fact. That's where hallucinations come from, and why grounding (RAG) and evals matter.
- Temperature and `max_tokens` are the sampling settings you'll use in chapter 05.
- More parameters mean more computation for every token, which is why bigger models are slower and cost more per token.

---

**How they fit together:**
1. Text is split into tokens and turned into vectors (§4).
2. A stack of Transformer blocks (§3) turns those vectors into a probability for the next token. The blocks are built from the neural-network basics of §1, and use attention to avoid the RNN problems of §2.
3. Training (§5) adjusts billions of weights by gradient descent until those predictions are good.
4. Generation samples from them, one token at a time.

**Further reading:**
- [Andrej Karpathy, *Neural Networks: Zero to Hero*](https://karpathy.ai/zero-to-hero.html): build all of this in code, from backpropagation to GPT and its tokenizer.
- [Jay Alammar, *The Illustrated Transformer*](https://jalammar.github.io/illustrated-transformer/)
- [Vaswani et al., *Attention Is All You Need* (2017)](https://arxiv.org/abs/1706.03762)
- [Hugging Face, *Byte-Pair Encoding tokenization*](https://huggingface.co/learn/llm-course/chapter6/5)
- [Kaplan et al., *Scaling Laws for Neural Language Models* (2020)](https://arxiv.org/abs/2001.08361)
- [Hoffmann et al., *Training Compute-Optimal Large Language Models* (2022)](https://arxiv.org/abs/2203.15556)

**Previous:** [← 03 · REST APIs, JSON and secrets](../03-rest-apis-and-secrets/README.md) · **Next:** 05 · Calling LLM APIs (coming soon)
