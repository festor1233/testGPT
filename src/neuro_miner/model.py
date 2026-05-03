"""Tiny neural network implemented with the Python standard library.

The project intentionally avoids heavy dependencies so it can run immediately in
GitHub Codespaces, classroom laptops and CI. The network is a one-hidden-layer
MLP trained with a simple temporal-difference target.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import random
from pathlib import Path
from typing import Sequence


@dataclass
class Prediction:
    value: float
    hidden: list[float]


class TinyMLP:
    def __init__(self, input_size: int = 8, hidden_size: int = 18, seed: int | None = None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        rng = random.Random(seed)
        self.w1 = [[rng.uniform(-0.25, 0.25) for _ in range(input_size)] for _ in range(hidden_size)]
        self.b1 = [0.0 for _ in range(hidden_size)]
        self.w2 = [rng.uniform(-0.25, 0.25) for _ in range(hidden_size)]
        self.b2 = 0.0

    def predict(self, x: Sequence[float]) -> Prediction:
        hidden = [math.tanh(sum(w * v for w, v in zip(row, x)) + bias) for row, bias in zip(self.w1, self.b1)]
        value = sum(w * h for w, h in zip(self.w2, hidden)) + self.b2
        return Prediction(value=value, hidden=hidden)

    def train_one(self, x: Sequence[float], target: float, lr: float = 0.025) -> float:
        pred = self.predict(x)
        error = pred.value - target
        for i in range(self.hidden_size):
            grad_w2 = error * pred.hidden[i]
            grad_hidden = error * self.w2[i] * (1 - pred.hidden[i] ** 2)
            self.w2[i] -= lr * grad_w2
            self.b1[i] -= lr * grad_hidden
            for j in range(self.input_size):
                self.w1[i][j] -= lr * grad_hidden * x[j]
        self.b2 -= lr * error
        return error * error

    def save(self, path: str | Path) -> None:
        payload = {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "w1": self.w1,
            "b1": self.b1,
            "w2": self.w2,
            "b2": self.b2,
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TinyMLP":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(payload["input_size"], payload["hidden_size"])
        model.w1 = payload["w1"]
        model.b1 = payload["b1"]
        model.w2 = payload["w2"]
        model.b2 = payload["b2"]
        return model
