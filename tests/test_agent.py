from neuro_miner.agent import NeuralSweeperAgent
from neuro_miner.game import Minesweeper


def test_agent_picks_hidden_cell():
    board = Minesweeper(seed=2)
    agent = NeuralSweeperAgent(seed=2)
    decision = agent.choose_cell(board)
    assert decision.cell in board.hidden_cells()


def test_learning_step_returns_loss():
    board = Minesweeper(seed=3)
    agent = NeuralSweeperAgent(seed=3)
    decision = agent.choose_cell(board)
    features = board.cell_features(*decision.cell)
    result = board.reveal(*decision.cell)
    loss = agent.learn_from_move(features, result.reward, board, result.done, lr=0.01)
    assert loss >= 0
