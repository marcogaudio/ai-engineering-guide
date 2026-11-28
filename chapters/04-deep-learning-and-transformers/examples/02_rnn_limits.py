"""Why RNNs struggle with long sequences: the first input fades away.

An RNN reads a sequence one step at a time and keeps a hidden state h:
    h_t = tanh(w * h_(t-1) + u * x_t)
To learn, it needs the gradient of the last state with respect to the first input: how much the
end of the sequence still "feels" its beginning. By the chain rule this gradient is a product
of one factor per step. With factors below 1 it vanishes; with factors above 1 it explodes.
Attention links the last position to the first one directly, in one step.

Run: uv run chapters/04-deep-learning-and-transformers/examples/02_rnn_limits.py
"""

import numpy as np


def first_input_influence(length: int, w: float, use_tanh: bool = True) -> float:
    """|d h_last / d x_first| for a 1-number RNN run over `length` random inputs."""
    xs = np.random.default_rng(0).normal(0, 1, length)
    h, grad = 0.0, 0.0
    for t, x in enumerate(xs):  # strictly one step after another: no parallelism possible
        pre = w * h + x  # u = 1
        h = float(np.tanh(pre)) if use_tanh else pre
        slope = 1 - h**2 if use_tanh else 1.0  # derivative of the activation at this step
        # Chain rule: step 1 receives x_first directly; every later step multiplies by w * slope.
        grad = slope if t == 0 else grad * w * slope
    return abs(grad)


if __name__ == "__main__":
    print("How much the last hidden state still depends on the FIRST input:\n")
    print(f"  {'length':>6}  {'tanh RNN, w=0.9':>16}  {'linear RNN, w=1.5':>18}  {'attention':>9}")
    for length in [1, 2, 5, 10, 20, 50]:
        vanishing = first_input_influence(length, w=0.9)
        exploding = first_input_influence(length, w=1.5, use_tanh=False)
        print(f"  {length:>6}  {vanishing:>16.2e}  {exploding:>18.2e}  {'1 step':>9}")

    print("\n- tanh RNN: the influence shrinks exponentially, so after a few dozen steps the")
    print("  first word barely affects anything and the model can't learn from it (vanishing).")
    print("- Linear RNN with w > 1: the same product blows up instead (exploding gradients).")
    print("- An RNN also needs `length` sequential steps; attention compares every position")
    print("  with every other one at once, so the first token is always one step away.")
