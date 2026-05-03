"""Learning agent that chooses Minesweeper cells with a neural value model."""
from __future__ import annotations

from dataclasses import dataclass
import random

from .game import Minesweeper, Coord
from .model import TinyMLP


@dataclass
class Decision:
    cell: Coord
    predicted_value: float
    exploratory: bool


class NeuralSweeperAgent:
    def __init__(self, model: TinyMLP | None = None, epsilon: float = 0.2, gamma: float = 0.92, seed: int | None = None):
        self.model = model or TinyMLP(seed=seed)
        self.epsilon = epsilon
        self.gamma = gamma
        self.rng = random.Random(seed)

    def choose_cell(self, board: Minesweeper) -> Decision:
        candidates = board.hidden_cells()
        if not candidates:
            return Decision((0, 0), 0.0, False)
        exploratory = self.rng.random() < self.epsilon or board.first_move
        if exploratory:
            cell = self.rng.choice(candidates)
            return Decision(cell, self.model.predict(board.cell_features(*cell)).value, True)
        scored = [(self.model.predict(board.cell_features(*cell)).value, cell) for cell in candidates]
        value, cell = max(scored, key=lambda item: item[0])
        return Decision(cell, value, False)

    def learn_from_move(self, before_features: list[float], reward: float, board: Minesweeper, done: bool, lr: float) -> float:
        if done:
            target = reward
        else:
            future = max((self.model.predict(board.cell_features(*cell)).value for cell in board.hidden_cells()), default=0.0)
            target = reward + self.gamma * future
        return self.model.train_one(before_features, target, lr=lr)
