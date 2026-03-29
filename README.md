# simple-maia3-inference

This repo provides a simple inference CLI and API for the [Maia3 model](maiachess.com).

It is possible (likely) that this repo will be deprecated later when the official Maia3 weights and inference wrapper come out, but for now they are provided as an onnx file and a typescript inference script that don't expose the full outputs of the model.

This repo contains the onnx weights (originally found [here](https://github.com/CSSLab/maia-platform-frontend/blob/44b3d4c82a45e5002460c36ae94d43881a8cd888/public/maia3/maia3_simplified.onnx)) and retains the same GPLv3 license as the original [Maia Chess repo](https://github.com/CSSLab/maia-platform-frontend).


# Installation

Clone the repo and install locally (it will be available on Pypi later):

```
git clone https://github.com/mcognetta/simple-maia3-inference
pip install -e simple-maia3-inference
```

# API

The API provides direct logit and probability access for single and batched inference.

```python
from maia3_inference import Maia3

maia = Maia3()
```

---

#### `probs(fen, elo_self, elo_oppo)`

Single-position inference. Returns move and WDL probabilities.

```python
move_probs, wdl = maia.probs(
    fen="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
    elo_self=1500,
    elo_oppo=1500,
)
# move_probs: {"e7e5": 0.312, "c7c5": 0.201, ...}  — legal moves, sorted by probability
# wdl:        (0.4823, 0.3011, 0.2166)              — (win, draw, loss) for side to move
```

---

#### `batch_probs(fens, elo_selfs, elo_oppos)`

Same as `probs` but accepts lists of positions. More efficient than calling `probs` in a loop.

```python
results = maia.batch_probs(fens=[...], elo_selfs=[...], elo_oppos=[...])
# results: list of (move_probs, wdl) tuples, one per input position
```

---

#### `logits(fen, elo_self, elo_oppo, mask_move_logits=True)`

Returns raw model output logits instead of probabilities. These can be optionally masked (via the `mask_move_logits` flag, which is set to `True` by default) to mask out invalid move probabilities.

```python
logits_move, logits_value = maia.logits(fen=..., elo_self=1500, elo_oppo=1500)
# logits_move:  np.ndarray shape (4352,) — one logit per candidate move
# logits_value: np.ndarray shape (3,)    — raw WDL logits
```

Pass `mask_move_logits=False` to skip setting illegal move logits to `-inf`.

---

#### `batch_logits(fens, elo_selfs, elo_oppos, mask_move_logits=True)`

Batched version of `logits`.

```python
logits_move, logits_value = maia.batch_logits(fens=[...], elo_selfs=[...], elo_oppos=[...])
# logits_move:  np.ndarray shape (N, 4352)
# logits_value: np.ndarray shape (N, 3)
```

---

The constructor accepts an optional `providers` list of [ONNX Runtime execution providers](https://onnxruntime.ai/docs/execution-providers/).
By default it auto-selects CUDA → CoreML → CPU. To force a specific provider:

```python
maia = Maia3(providers=["CPUExecutionProvider"])
```


# CLI

The CLI allows for a quick, single inference call on a FEN and self/opponent rating.

```
usage: simple-maia3-inference [-h] --fen FEN [--elo-self ELO] [--elo-oppo ELO]

Maia3 chess move probability inference

options:
  -h, --help      show this help message and exit
  --fen FEN       FEN string for single-position inference
  --elo-self ELO  Elo rating of the side to move (default: 1500)
  --elo-oppo ELO  Elo rating of the opponent (default: 1500)
```


For example:

```
simple-maia3-inference --fen '8/8/7B/1p3kpp/p1b5/2P2KP1/1P6/8 b - - 3 47' --elo-self 1569 --elo-oppo 1579

FEN:  8/8/7B/1p3kpp/p1b5/2P2KP1/1P6/8 b - - 3 47
Elo:  1569 (self) vs 1579 (opponent)
WDL:  0.5693 / 0.3422 / 0.0884
Move probabilities (17 legal moves):
  g5g4   g4+    44.46%  #################
  h5h4   h4     17.69%  #######
  c4d5   Bd5+   17.69%  #######
  f5g6   Kg6    11.42%  ####
  c4b3   Bb3     4.20%  #
  c4d3   Bd3     2.80%  #
  c4f1   Bf1     0.75%  
  c4e6   Be6     0.26%  
  c4f7   Bf7     0.20%  
  a4a3   a3      0.12%  
  c4e2   Be2+    0.10%  
  c4a2   Ba2     0.10%  
  f5e5   Ke5     0.09%  
  f5f6   Kf6     0.05%  
  c4g8   Bg8     0.05%  
  b5b4   b4      0.02%  
  f5e6   Ke6     0.01%  
```
