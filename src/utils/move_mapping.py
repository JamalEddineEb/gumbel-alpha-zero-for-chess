# utils/move_mapping.py
# Defines a stable global action space and helpers to map between chess.Move <-> index.
# Action space: all from->to pairs on 8x8 board (64*64 = 4096).

from __future__ import annotations
import json
from typing import Dict, List

import chess
import numpy as np


class MoveMapping:
    """
    Global, deterministic mapping between chess moves (uci strings) and action indices.
    - If INCLUDE_PROMOTIONS=False: index = from_sq * 64 + to_sq  in [0, 4095]
    - If INCLUDE_PROMOTIONS=True: promotion moves get distinct indices appended after base 4096 range.
    Provides batch helpers for vectorized gathering.
    """
    def __init__(self):
        self.move_to_idx: Dict[str, int] = {}
        self.idx_to_move: Dict[int, str] = {}
        self._build()

    @property
    def num_actions(self) -> int:
        return len(self.idx_to_move)

    def _build(self):
        # Base mapping: 64x64 from->to pairs
        idx = 0
        for from_sq in range(64):
            for to_sq in range(64):
                # base UCI without promotion (e.g., e2e4)
                uci = chess.SQUARE_NAMES[from_sq] + chess.SQUARE_NAMES[to_sq]
                self.move_to_idx[uci] = idx
                self.idx_to_move[idx] = uci
                idx += 1

        # Promotion moves
        promotions = ['q', 'r', 'b', 'n']
        
        # White promotions: Rank 7 -> Rank 8
        for f in range(8):
            from_sq = chess.square(f, 6)
            targets = [f]
            if f > 0: targets.append(f - 1)
            if f < 7: targets.append(f + 1)
            
            for t in targets:
                to_sq = chess.square(t, 7)
                base_uci = chess.SQUARE_NAMES[from_sq] + chess.SQUARE_NAMES[to_sq]
                for p in promotions:
                    uci = base_uci + p
                    self.move_to_idx[uci] = idx
                    self.idx_to_move[idx] = uci
                    idx += 1

        # Black promotions: Rank 2 -> Rank 1
        for f in range(8):
            from_sq = chess.square(f, 1)
            targets = [f]
            if f > 0: targets.append(f - 1)
            if f < 7: targets.append(f + 1)
            
            for t in targets:
                to_sq = chess.square(t, 0)
                base_uci = chess.SQUARE_NAMES[from_sq] + chess.SQUARE_NAMES[to_sq]
                for p in promotions:
                    uci = base_uci + p
                    self.move_to_idx[uci] = idx
                    self.idx_to_move[idx] = uci
                    idx += 1


    def get_index(self, move_uci: str) -> int:
        return self.move_to_idx[move_uci]

    def get_move_uci(self, idx: int) -> str:
        return self.idx_to_move[idx]

    def batch_moves_to_indices(self, moves: List[chess.Move]) -> np.ndarray:
        # Vectorized-friendly conversion; returns int array
        return np.fromiter((self.move_to_idx[m.uci()] for m in moves), dtype=np.int32, count=len(moves))

    def batch_indices_to_moves(self, indices: np.ndarray) -> List[str]:
        # Returns UCI strings; caller can convert to chess.Move(board) if needed
        return [self.idx_to_move[int(i)] for i in indices]

    def save_json(self, path: str):
        payload = {
            "move_to_idx": self.move_to_idx,
            "idx_to_move": {str(k): v for v, k in self.move_to_idx.items()}  
        }
        # The above reverse is redundant; instead persist both bijections explicitly:
        payload = {
            "move_to_idx": self.move_to_idx,
            "idx_to_move": {int(k): v for k, v in self.idx_to_move.items()}
        }
        # Ensure int keys are serialized properly
        payload["idx_to_move"] = {str(k): v for k, v in payload["idx_to_move"].items()}
        with open(path, "w") as f:
            json.dump(payload, f)

    @staticmethod
    def load_json(path: str) -> "MoveMapping":
        with open(path, "r") as f:
            data = json.load(f)
        mm = MoveMapping.__new__(MoveMapping)  # bypass __init__ build
        mm.move_to_idx = {str(k): int(v) for k, v in data["move_to_idx"].items()}
        # idx_to_move keys were saved as strings
        mm.idx_to_move = {int(k): str(v) for k, v in data["idx_to_move"].items()}
        # Basic consistency checks
        assert len(mm.move_to_idx) == len(mm.idx_to_move)
        # Bijective check on a small sample
        for k in list(mm.move_to_idx.keys())[:50]:
            idx = mm.move_to_idx[k]
            assert mm.idx_to_move[idx] == k
        return mm

# Singleton-style accessor to avoid rebuilding repeatedly
_global_mapping: MoveMapping | None = None

def get_global_mapping() -> MoveMapping:
    global _global_mapping
    if _global_mapping is None:
        _global_mapping = MoveMapping()
    return _global_mapping

# Convenience helpers used by search/training

def legal_indices_from_moves(moves: List[chess.Move]) -> np.ndarray:
    mm = get_global_mapping()
    return mm.batch_moves_to_indices(moves)

def gather_legal_logits(policy_logits: np.ndarray, legal_indices: np.ndarray) -> np.ndarray:
    # policy_logits: shape (A,) global; legal_indices: (L,)
    return policy_logits[legal_indices]

def masked_softmax_legal(policy_logits: np.ndarray, legal_indices: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    # Returns probabilities over legal_indices summing to 1
    z = policy_logits[legal_indices].astype(np.float64)
    z -= np.max(z)
    expz = np.exp(z)
    s = expz.sum()
    if not np.isfinite(s) or s <= 0.0:
        return np.full(len(legal_indices), 1.0 / max(1, len(legal_indices)), dtype=np.float64)
    p = expz / s
    p = np.maximum(p, eps)
    p /= p.sum()
    return p
