"""Command line trainer for Neuro Miner."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from .agent import NeuralSweeperAgent
from .game import Minesweeper
from .model import TinyMLP


@dataclass
class TrainingStats:
    games: int = 0
    wins: int = 0
    total_moves: int = 0
    total_loss: float = 0.0

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0


def run_episode(agent: NeuralSweeperAgent, width: int, height: int, mines: int, lr: float, max_moves: int, seed: int | None = None) -> tuple[bool, int, float]:
    board = Minesweeper(width, height, mines, seed=seed)
    loss = 0.0
    moves = 0
    while not board.done and moves < max_moves:
        decision = agent.choose_cell(board)
        features = board.cell_features(*decision.cell)
        result = board.reveal(*decision.cell)
        loss += agent.learn_from_move(features, result.reward, board, result.done, lr=lr)
        moves += 1
    return board.won, moves, loss


def train(args: argparse.Namespace) -> TrainingStats:
    model = TinyMLP(seed=args.seed)
    agent = NeuralSweeperAgent(model=model, epsilon=args.epsilon, gamma=args.gamma, seed=args.seed)
    stats = TrainingStats()
    for episode in range(1, args.episodes + 1):
        agent.epsilon = max(args.min_epsilon, args.epsilon * (args.decay ** (episode - 1)))
        won, moves, loss = run_episode(agent, args.width, args.height, args.mines, args.lr, args.max_moves, seed=(args.seed or 0) + episode)
        stats.games += 1
        stats.wins += int(won)
        stats.total_moves += moves
        stats.total_loss += loss
        if episode % args.report_every == 0 or episode == 1:
            print(f"episode={episode:5d} win_rate={stats.win_rate:.3f} epsilon={agent.epsilon:.3f} avg_moves={stats.total_moves / stats.games:.1f} loss={loss:.3f}")
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        model.save(args.output)
        print(f"saved model -> {args.output}")
    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a neural Minesweeper agent.")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--width", type=int, default=9)
    parser.add_argument("--height", type=int, default=9)
    parser.add_argument("--mines", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.025)
    parser.add_argument("--gamma", type=float, default=0.92)
    parser.add_argument("--epsilon", type=float, default=0.35)
    parser.add_argument("--min-epsilon", type=float, default=0.04)
    parser.add_argument("--decay", type=float, default=0.995)
    parser.add_argument("--max-moves", type=int, default=250)
    parser.add_argument("--report-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="models/neuro-miner.json")
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
