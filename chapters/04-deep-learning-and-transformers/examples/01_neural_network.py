"""A tiny neural network learns XOR, with the forward and backward pass written by hand.

XOR outputs 1 when exactly one of its two inputs is 1. No straight line separates the 1s from
the 0s, so a model with no hidden layer can't learn it. One hidden layer with a non-linear
activation can.

Run: uv run chapters/04-deep-learning-and-transformers/examples/01_neural_network.py
"""

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]

X: Array = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)  # 4 examples, 2 inputs each
Y: Array = np.array([[0], [1], [1], [0]], dtype=float)  # the right answers (labels)


def sigmoid(z: Array) -> Array:
    """Activation function: squashes any number into the range 0..1."""
    return 1 / (1 + np.exp(-z))


def mse(pred: Array, target: Array) -> float:
    """Loss: the average squared distance from the right answers. 0 means perfect."""
    return float(np.mean((pred - target) ** 2))


def train(hidden: int, steps: int = 4000, lr: float = 2.0) -> Array:
    """Train a 2 -> hidden -> 1 network with gradient descent and return its predictions."""
    rng = np.random.default_rng(0)
    # Parameters: one weight per connection and one bias per neuron, starting small and random.
    w1, b1 = rng.normal(0, 1, (2, hidden)), np.zeros(hidden)
    w2, b2 = rng.normal(0, 1, (hidden, 1)), np.zeros(1)
    for step in range(steps + 1):
        # Forward pass: inputs -> hidden layer -> output. Each neuron computes a weighted sum
        # of its inputs, adds its bias, and applies the activation function.
        h = sigmoid(X @ w1 + b1)  # (4, hidden)
        out = sigmoid(h @ w2 + b2)  # (4, 1): the predictions
        if step % 1000 == 0:
            print(f"  step {step:4}: loss {mse(out, Y):.4f}")
        # Backward pass (backpropagation): the chain rule, applied from the loss back to every
        # weight. Each line reuses the one above it, which is what makes it cheap.
        d_out = 2 * (out - Y) / len(X) * out * (1 - out)  # through the loss and output sigmoid
        d_h = d_out @ w2.T * h * (1 - h)  # back through w2 and the hidden sigmoid
        # Gradient descent: move every parameter a small step against its gradient.
        w2 -= lr * h.T @ d_out
        b2 -= lr * d_out.sum(axis=0)
        w1 -= lr * X.T @ d_h
        b1 -= lr * d_h.sum(axis=0)
    return out


def train_without_hidden_layer(steps: int = 4000, lr: float = 2.0) -> Array:
    """The same training loop with no hidden layer: output = sigmoid(X @ w + b)."""
    w, b = np.zeros((2, 1)), np.zeros(1)
    for _ in range(steps):
        out = sigmoid(X @ w + b)
        d_out = 2 * (out - Y) / len(X) * out * (1 - out)
        w -= lr * X.T @ d_out
        b -= lr * d_out.sum(axis=0)
    return sigmoid(X @ w + b)


if __name__ == "__main__":
    print("With one hidden layer of 4 neurons (2*4 + 4 + 4*1 + 1 = 17 parameters):")
    pred = train(hidden=4)
    for x, p, y in zip(X, pred, Y, strict=True):
        print(f"  input {x.astype(int)} -> {p[0]:.2f}   (right answer {y[0]:.0f})")

    flat = train_without_hidden_layer()
    print(f"\nWithout a hidden layer: loss {mse(flat, Y):.4f}, predictions {flat.ravel().round(2)}")
    print("  it can only draw one straight line, so the best it can do is answer ~0.5 for all.")
