import chess
import numpy as np

from .constants import ALL_MOVES_MAIA3

_piece_to_idx = {p: i for i, p in enumerate(["P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"])}

def board_to_maia3_tokens(fen: str) -> np.ndarray:

    tensor = np.zeros(64 * 12, dtype=np.float32)

    piece_placement = fen.split(" ")[0]
    rows = piece_placement.split("/")

    for rank in range(8):
        row = 7 - rank
        file = 0
        for char in rows[rank]:
            if char.isdigit():
                file += int(char)
            else:
                piece_idx = _piece_to_idx[char]
                square = row * 8 + file
                tensor[square * 12 + piece_idx] = 1.0
                file += 1

    return tensor


def preprocess_maia3(fen: str) -> tuple[np.ndarray, np.ndarray]:
    board = chess.Board(fen)
    if board.turn == chess.BLACK:
        board = board.mirror()

    board_tokens = board_to_maia3_tokens(board.fen())

    num_moves = len(ALL_MOVES_MAIA3)
    legal_moves = np.zeros(num_moves, dtype=np.float32)
    for move in board.legal_moves:
        move_idx = ALL_MOVES_MAIA3[move.uci()]
        legal_moves[move_idx] = 1.0

    return board_tokens, legal_moves


def mirror_square(square: str) -> str:
    return square[0] + str(9 - int(square[1]))


def mirror_move(move_uci: str) -> str:
    promotion = move_uci[4:] if len(move_uci) > 4 else ""
    return mirror_square(move_uci[:2]) + mirror_square(move_uci[2:4]) + promotion
