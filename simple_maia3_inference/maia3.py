import argparse
from pathlib import Path

import chess

import numpy as np
import onnxruntime as ort

from .constants import ALL_MOVES_MAIA3_REVERSED
from .tensor import mirror_move, preprocess_maia3

_MODEL_PATH = Path(__file__).parent / "maia3_simplified.onnx"


def _get_providers() -> list[str]:
    """
    Auto-selects the best available execution provider.
    Override by passing providers= explicitly to Maia3.

    Priority: CUDA (Nvidia) > CoreML (Apple Metal) > CPU
    For CoreML, install: pip install onnxruntime-silicon
    For CUDA, install:   pip install onnxruntime-gpu
    """
    available = set(ort.get_available_providers())
    for provider in ("CUDAExecutionProvider", "CoreMLExecutionProvider"):
        if provider in available:
            return [provider, "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _compute_probs(
    fen: str,
    logits_move: np.ndarray,
    logits_value: np.ndarray,
    legal_moves: np.ndarray,
) -> tuple[dict[str, float], tuple[float, float, float]]:
    """
    Converts raw maia3 ONNX outputs into a move probability dict and WDL tuple.

    Args:
        fen:          FEN string of the position; used to determine the side to move.
        logits_move:  Raw move logits from the ONNX model, shape (4352,).
        logits_value: Raw WDL logits from the ONNX model, shape (3,) — [win, draw, loss].
        legal_moves:  Binary mask over the 4352 move indices; 1 = legal, 0 = illegal.

    Returns:
        move_probs: move → probability dict (legal moves only, sorted descending).
        wdl:        (win, draw, loss) probabilities from the side-to-move's perspective.
    """
    wdl = logits_value - logits_value.max()
    exp_wdl = np.exp(wdl)
    exp_wdl /= exp_wdl.sum()

    black_flag = fen.split(" ")[1] == "b"
    if black_flag:
        exp_wdl = exp_wdl[::-1]
    wdl_probs = tuple(round(float(p), 4) for p in exp_wdl)

    legal_indices = np.where(legal_moves > 0)[0]
    move_ucis = []
    for idx in legal_indices:
        uci = ALL_MOVES_MAIA3_REVERSED[int(idx)]
        if black_flag:
            uci = mirror_move(uci)
        move_ucis.append(uci)

    legal_logits = logits_move[legal_indices]
    legal_logits = legal_logits - legal_logits.max()
    exp_logits = np.exp(legal_logits)
    probs = exp_logits / exp_logits.sum()

    move_probs = {uci: float(probs[i]) for i, uci in enumerate(move_ucis)}
    move_probs = dict(sorted(move_probs.items(), key=lambda x: x[1], reverse=True))

    return move_probs, wdl_probs


class Maia3:
    def __init__(self, providers: list[str] | None = None):
        """
        Args:
            providers:  ORT execution providers. Defaults to auto-detected best available.
        """
        self.session = ort.InferenceSession(
            str(_MODEL_PATH),
            providers=providers or _get_providers(),
        )

    def _run(
        self,
        boards: list[str],
        elo_selfs: list[float],
        elo_oppos: list[float],
    ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
        """
        Shared preprocessing and inference for all public methods.

        Args:
            boards:    List of FEN strings, one per position.
            elo_selfs: Elo rating of the side to move for each position.
            elo_oppos: Elo rating of the opponent for each position.
        """
        tokens_list, legal_moves_list = [], []
        for fen in boards:
            tokens, legal_moves = preprocess_maia3(fen)
            tokens_list.append(tokens)
            legal_moves_list.append(legal_moves)

        n = len(boards)
        feeds = {
            "tokens": np.stack(tokens_list).reshape(n, 64, 12),
            "elo_self": np.array(elo_selfs, dtype=np.float32),
            "elo_oppo": np.array(elo_oppos, dtype=np.float32),
        }
        logits_move, logits_value = self.session.run(
            ["logits_move", "logits_value"], feeds
        )
        return logits_move, logits_value, np.stack(legal_moves_list)

    # ------------------------------------------------------------------
    # Raw logits
    # ------------------------------------------------------------------

    def logits(
        self, fen: str, elo_self: float, elo_oppo: float, mask_move_logits: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Single-position inference returning raw logits.

        Args:
            fen:               FEN string of the position to evaluate.
            elo_self:          Elo rating of the side to move.
            elo_oppo:          Elo rating of the opponent.
            mask_move_logits:  If True, sets logits for illegal moves to -inf (default True).

        Returns:
            logits_move:  shape (4352,) — one logit per candidate move
            logits_value: shape (3,)    — WDL logits
        """
        lm, lv, legal_moves = self._run([fen], [elo_self], [elo_oppo])
        lm, lv, legal_moves = lm[0], lv[0], legal_moves[0]
        if mask_move_logits:
            lm[legal_moves == 0.0] = -np.inf
        return lm, lv

    def batch_logits(
        self,
        fens: list[str],
        elo_selfs: list[float],
        elo_oppos: list[float],
        mask_move_logits: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Batch inference returning raw logits.

        Args:
            fens:              List of FEN strings, one per position.
            elo_selfs:         Elo rating of the side to move for each position.
            elo_oppos:         Elo rating of the opponent for each position.
            mask_move_logits:  If True, sets logits for illegal moves to -inf (default True).

        Returns:
            logits_move:  shape (N, 4352)
            logits_value: shape (N, 3)
        """
        lm, lv, legal_moves = self._run(fens, elo_selfs, elo_oppos)
        if mask_move_logits:
            lm[legal_moves == 0.0] = -np.inf
        return lm, lv

    # ------------------------------------------------------------------
    # Move probability dicts
    # ------------------------------------------------------------------

    def probs(
        self, fen: str, elo_self: float, elo_oppo: float
    ) -> tuple[dict[str, float], tuple[float, float, float]]:
        """
        Single-position inference returning move probabilities and WDL.

        Args:
            fen:      FEN string of the position to evaluate.
            elo_self: Elo rating of the side to move.
            elo_oppo: Elo rating of the opponent.

        Returns:
            move_probs: move → probability dict (legal moves only, sorted descending).
            wdl:        (win, draw, loss) probabilities from the side-to-move's perspective.
        """
        lm, lv, legal = self._run([fen], [elo_self], [elo_oppo])
        return _compute_probs(fen, lm[0], lv[0], legal[0])

    def batch_probs(
        self,
        fens: list[str],
        elo_selfs: list[float],
        elo_oppos: list[float],
    ) -> list[tuple[dict[str, float], tuple[float, float, float]]]:
        """
        Batch inference returning move probabilities and WDL per position.

        Args:
            fens:      List of FEN strings, one per position.
            elo_selfs: Elo rating of the side to move for each position.
            elo_oppos: Elo rating of the opponent for each position.

        Returns:
            List of (move_probs, wdl) tuples, one per input position.
            move_probs: move → probability dict (legal moves only, sorted descending).
            wdl:        (win, draw, loss) probabilities from the side-to-move's perspective.
        """
        lm, lv, legal = self._run(fens, elo_selfs, elo_oppos)
        return [
            _compute_probs(fens[i], lm[i], lv[i], legal[i])
            for i in range(len(fens))
        ]


# ----------------------------------------------------------------------
# CLI helpers
# ----------------------------------------------------------------------

def _run_single(args, maia: Maia3) -> None:
    lm, lv, legal = maia._run([args.fen], [args.elo_self], [args.elo_oppo])
    policy, wdl = _compute_probs(args.fen, lm[0], lv[0], legal[0])

    print(f"\nFEN:  {args.fen}")
    print(f"Elo:  {args.elo_self:.0f} (self) vs {args.elo_oppo:.0f} (opponent)")
    print(f"WDL:  {wdl[0]:.4f} / {wdl[1]:.4f} / {wdl[2]:.4f}")
    print(f"Move probabilities ({len(policy)} legal moves):")
    board = chess.Board(args.fen)
    for move, prob in policy.items():
        bar = "#" * int(prob * 40)
        print(f"  {move:6s} {board.san(chess.Move.from_uci(move)):6s} {prob:6.2%}  {bar}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="simple-maia3-inference",
        description="Maia3 chess move probability inference",
    )

    parser.add_argument("--fen", required=True, metavar="FEN", help="FEN string for single-position inference")
    parser.add_argument(
        "--elo-self",
        type=float,
        default=1500.0,
        metavar="ELO",
        help="Elo rating of the side to move (default: 1500)",
    )
    parser.add_argument(
        "--elo-oppo",
        type=float,
        default=1500.0,
        metavar="ELO",
        help="Elo rating of the opponent (default: 1500)",
    )
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()
    maia = Maia3()
    _run_single(args, maia)


if __name__ == "__main__":
    main()
