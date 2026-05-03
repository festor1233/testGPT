from neuro_miner.game import Minesweeper, CellState


def test_first_click_is_safe_and_reveals_cells():
    board = Minesweeper(seed=1)
    result = board.reveal(0, 0)
    assert not result.done
    assert (0, 0) not in board.mines
    assert board.state[0][0] == CellState.REVEALED


def test_board_can_render():
    board = Minesweeper(seed=1)
    text = board.render()
    assert "□" in text
    assert "0" in text
