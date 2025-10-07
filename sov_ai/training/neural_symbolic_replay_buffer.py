"""Replay buffer for challenging training samples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

from collections import deque


@dataclass
class ReplayItem:
    sample: Dict[str, object]
    difficulty: float
    metadata: Optional[Dict[str, object]] = None


class NeuralSymbolicReplayBuffer:
    """Prioritise difficult samples for neural-symbolic rehearsal."""

    def __init__(
        self,
        capacity: int = 128,
        min_difficulty: float = 0.0,
        temperature: float = 1.0,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = capacity
        self._min_difficulty = min_difficulty
        self._temperature = temperature
        self._items: Deque[ReplayItem] = deque(maxlen=capacity)

    def add(self, sample: Dict[str, object], difficulty: float, metadata: Optional[Dict[str, object]] = None) -> None:
        if difficulty < self._min_difficulty:
            return
        item = ReplayItem(sample=sample, difficulty=float(difficulty), metadata=metadata)
        self._items.append(item)

    def _sorted_items(self) -> List[ReplayItem]:
        return sorted(self._items, key=lambda item: item.difficulty, reverse=True)

    def sample(self, batch_size: int, strategy: str = "hybrid") -> List[ReplayItem]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        items = self._sorted_items()
        if not items:
            return []
        if strategy == "topk":
            return items[:batch_size]
        if strategy == "stochastic":
            weights = [item.difficulty ** self._temperature for item in items]
            total = sum(weights)
            if total <= 0:
                return items[:batch_size]
            probabilities = [w / total for w in weights]
            selected: List[ReplayItem] = []
            for _ in range(min(batch_size, len(items))):
                cumulative = 0.0
                from random import random

                r = random()
                for idx, prob in enumerate(probabilities):
                    cumulative += prob
                    if r <= cumulative:
                        selected.append(items[idx])
                        break
            return selected
        # Hybrid: mix top-k and stochastic sampling.
        half = max(1, batch_size // 2)
        top = items[:half]
        remainder = batch_size - len(top)
        if remainder <= 0:
            return top
        stochastic = self.sample(remainder, strategy="stochastic")
        combined = top + [item for item in stochastic if item not in top]
        return combined[:batch_size]

    def __len__(self) -> int:
        return len(self._items)

    def state_dict(self) -> Dict[str, object]:
        return {
            "capacity": self._capacity,
            "min_difficulty": self._min_difficulty,
            "temperature": self._temperature,
            "items": [
                {
                    "sample": item.sample,
                    "difficulty": item.difficulty,
                    "metadata": item.metadata,
                }
                for item in self._items
            ],
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self._capacity = int(state.get("capacity", self._capacity))
        self._min_difficulty = float(state.get("min_difficulty", self._min_difficulty))
        self._temperature = float(state.get("temperature", self._temperature))
        self._items = deque(maxlen=self._capacity)
        for raw in state.get("items", []):
            self.add(
                sample=dict(raw.get("sample", {})),
                difficulty=float(raw.get("difficulty", 0.0)),
                metadata=raw.get("metadata"),
            )
