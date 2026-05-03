"""Watch a trained agent play a board in the terminal."""
from __future__ import annotations

import argparse
import time

from .agent import NeuralSweeperAgent
from .game import Minesweeper
from .model import TinyMLP


def main() -> None:
    parser = argparse.ArgumentParser(description="Let the neural agent play Minesweeper.")
    parser.add_argument("--model", default="models/neuro-miner.json")
    parser.add_argument("--width", type=int, default=9)
    parser.add_argument("--height", type=int, default=9)
    parser.add_argument("--mines", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delay", type=float, default=0.05)
    args = parser.parse_args()

    board = Minesweeper(args.width, args.height, args.mines, seed=args.seed)
    agent = NeuralSweeperAgent(TinyMLP.load(args.model), epsilon=0.0, seed=args.seed)
    print(board.render())
    while not board.done:
        decision = agent.choose_cell(board)
        result = board.reveal(*decision.cell)
        print("\n" + "=" * 36)
        print(f"move={board.moves} cell={decision.cell} value={decision.predicted_value:.3f} -> {result.message}")
        print(board.render(reveal_mines=result.done))
        time.sleep(args.delay)
    print("\n🏆 Victory!" if board.won else "\n💥 Defeat. Train longer and try again.")


if __name__ == "__main__":
    main()
