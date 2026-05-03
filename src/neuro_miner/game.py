"""Core Minesweeper environment used by the learning agent."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import random
from typing import Iterable, Tuple

Coord = Tuple[int, int]


class CellState(str, Enum):
    HIDDEN = "hidden"
    REVEALED = "revealed"
    FLAGGED = "flagged"


@dataclass(frozen=True)
class StepResult:
    reward: float
    done: bool
    won: bool
    message: str


class Minesweeper:
    """A compact, deterministic Minesweeper board.

    The first reveal is always safe: mines are placed after the first click and
    exclude the clicked cell plus its neighbours. This makes training fairer and
    mirrors many desktop Minesweeper implementations.
    """

    def __init__(self, width: int = 9, height: int = 9, mines: int = 10, seed: int | None = None):
        if width < 4 or height < 4:
            raise ValueError("Board must be at least 4x4.")
        if mines <= 0 or mines >= width * height - 9:
            raise ValueError("Mine count must leave enough safe cells around the first click.")
        self.width = width
        self.height = height
        self.mines_count = mines
        self._rng = random.Random(seed)
        self.reset()

    def reset(self) -> None:
        self.mines: set[Coord] = set()
        self.values: list[list[int]] = [[0 for _ in range(self.width)] for _ in range(self.height)]
        self.state: list[list[CellState]] = [[CellState.HIDDEN for _ in range(self.width)] for _ in range(self.height)]
        self.first_move = True
        self.done = False
        self.won = False
        self.moves = 0

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.height and 0 <= col < self.width

    def neighbours(self, row: int, col: int) -> Iterable[Coord]:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if self.in_bounds(nr, nc):
                    yield nr, nc

    def _place_mines(self, safe: Coord) -> None:
        forbidden = {safe, *self.neighbours(*safe)}
        candidates = [(r, c) for r in range(self.height) for c in range(self.width) if (r, c) not in forbidden]
        self.mines = set(self._rng.sample(candidates, self.mines_count))
        for r in range(self.height):
            for c in range(self.width):
                self.values[r][c] = -1 if (r, c) in self.mines else sum((n in self.mines) for n in self.neighbours(r, c))

    def hidden_cells(self) -> list[Coord]:
        return [(r, c) for r in range(self.height) for c in range(self.width) if self.state[r][c] == CellState.HIDDEN]

    def visible_safe_count(self) -> int:
        return sum(self.state[r][c] == CellState.REVEALED for r in range(self.height) for c in range(self.width))

    def is_solved(self) -> bool:
        return self.visible_safe_count() == self.width * self.height - self.mines_count

    def reveal(self, row: int, col: int) -> StepResult:
        if self.done:
            return StepResult(-0.2, True, self.won, "Game is already over.")
        if not self.in_bounds(row, col):
            return StepResult(-0.2, False, False, "Move is outside the board.")
        if self.state[row][col] == CellState.FLAGGED:
            return StepResult(-0.15, False, False, "Cell is flagged.")
        if self.state[row][col] == CellState.REVEALED:
            return StepResult(-0.05, False, False, "Cell was already revealed.")

        if self.first_move:
            self._place_mines((row, col))
            self.first_move = False

        self.moves += 1
        if (row, col) in self.mines:
            self.state[row][col] = CellState.REVEALED
            self.done = True
            self.won = False
            return StepResult(-1.0, True, False, "Boom! Mine hit.")

        revealed = self._flood_reveal(row, col)
        if self.is_solved():
            self.done = True
            self.won = True
            return StepResult(1.5, True, True, "Board solved.")
        return StepResult(0.04 * revealed, False, False, f"Revealed {revealed} safe cell(s).")

    def toggle_flag(self, row: int, col: int) -> StepResult:
        if self.done:
            return StepResult(-0.1, True, self.won, "Game is already over.")
        if not self.in_bounds(row, col) or self.state[row][col] == CellState.REVEALED:
            return StepResult(-0.1, False, False, "Cannot flag this cell.")
        self.state[row][col] = CellState.FLAGGED if self.state[row][col] == CellState.HIDDEN else CellState.HIDDEN
        return StepResult(0.0, False, False, "Flag toggled.")

    def _flood_reveal(self, row: int, col: int) -> int:
        queue = [(row, col)]
        revealed = 0
        while queue:
            r, c = queue.pop()
            if self.state[r][c] == CellState.REVEALED:
                continue
            self.state[r][c] = CellState.REVEALED
            revealed += 1
            if self.values[r][c] == 0:
                for nr, nc in self.neighbours(r, c):
                    if self.state[nr][nc] == CellState.HIDDEN and (nr, nc) not in self.mines:
                        queue.append((nr, nc))
        return revealed

    def cell_features(self, row: int, col: int) -> list[float]:
        """Feature vector for an unrevealed candidate cell."""
        hidden_neigh = flagged_neigh = revealed_neigh = number_sum = 0
        for nr, nc in self.neighbours(row, col):
            cell_state = self.state[nr][nc]
            if cell_state == CellState.HIDDEN:
                hidden_neigh += 1
            elif cell_state == CellState.FLAGGED:
                flagged_neigh += 1
            else:
                revealed_neigh += 1
                number_sum += max(self.values[nr][nc], 0)
        total_hidden = len(self.hidden_cells()) / (self.width * self.height)
        return [
            row / max(self.height - 1, 1),
            col / max(self.width - 1, 1),
            hidden_neigh / 8,
            flagged_neigh / 8,
            revealed_neigh / 8,
            number_sum / 24,
            total_hidden,
            self.mines_count / (self.width * self.height),
        ]

    def render(self, reveal_mines: bool = False) -> str:
        header = "   " + " ".join(str(c) for c in range(self.width))
        rows = [header]
        for r in range(self.height):
            chars: list[str] = []
            for c in range(self.width):
                state = self.state[r][c]
                if reveal_mines and (r, c) in self.mines:
                    chars.append("*")
                elif state == CellState.HIDDEN:
                    chars.append("□")
                elif state == CellState.FLAGGED:
                    chars.append("⚑")
                elif self.values[r][c] == 0:
                    chars.append("·")
                else:
                    chars.append(str(self.values[r][c]))
            rows.append(f"{r:2} " + " ".join(chars))
        return "\n".join(rows)
