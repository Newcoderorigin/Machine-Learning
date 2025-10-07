"""Symbolic learning loop implementation."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch

SymbolicSignal = Dict[str, float]


@dataclass
class LearningState:
    """Container for model parameters and metadata."""

    weights: torch.Tensor
    bias: torch.Tensor
    interactions: int = 0


class SymbolicLearner:
    """Shallow reinforcement learner with symbolic adjustments."""

    def __init__(self, feature_dim: int = 256, lr: float = 0.05) -> None:
        self._feature_dim = feature_dim
        self._lr = lr
        self._state = LearningState(
            weights=torch.zeros(feature_dim, dtype=torch.float32),
            bias=torch.zeros(1, dtype=torch.float32),
        )

    @staticmethod
    def _hash_token(token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
        return int.from_bytes(digest, byteorder="big", signed=False)

    def _encode(self, text: str) -> torch.Tensor:
        vector = torch.zeros(self._feature_dim, dtype=torch.float32)
        if not text:
            return vector
        for token in text.lower().split():
            idx = self._hash_token(token) % self._feature_dim
            vector[idx] += 1.0
        return vector

    def score(self, text: str) -> float:
        features = self._encode(text)
        activation = torch.dot(self._state.weights, features) + self._state.bias
        return float(torch.tanh(activation))

    def reinforce(self, text: str, reward: Optional[float]) -> None:
        if reward is None:
            return
        reward_tensor = torch.tensor(reward, dtype=torch.float32)
        features = self._encode(text)
        grad = (1 - torch.tanh(torch.dot(self._state.weights, features)) ** 2) * reward_tensor
        self._state.weights += self._lr * grad * features
        self._state.bias += self._lr * grad
        self._state.interactions += 1

    def symbolic_adjust(self, feedback: Optional[str], magnitude: float = 0.1) -> None:
        if not feedback:
            return
        features = self._encode(feedback)
        self._state.weights += magnitude * features
        self._state.interactions += 1

    def export_state(self) -> Dict[str, float]:
        norm = torch.linalg.norm(self._state.weights).item()
        return {
            "feature_dim": float(self._feature_dim),
            "weights_norm": norm,
            "bias": float(self._state.bias.item()),
            "interactions": float(self._state.interactions),
        }


def analyze_text(text: str, learner: SymbolicLearner) -> SymbolicSignal:
    """Produce a symbolic analysis signal for the provided text."""

    score = learner.score(text)
    length = len(text.split())
    entropy = -score * math.log2(abs(score) + 1e-6)
    return {"score": score, "length": float(length), "entropy": entropy}


__all__ = ["SymbolicLearner", "LearningState", "SymbolicSignal", "analyze_text"]
