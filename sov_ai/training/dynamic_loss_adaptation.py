"""Adaptive weighting for multiple loss terms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, MutableMapping, Optional, Tuple

from ._compat import (
    BackendLiteral,
    add_values,
    detect_backend,
    scale_value,
    to_scalar,
)


@dataclass
class DynamicLossAdaptation:
    """Maintain adaptive weights for multi-objective training losses."""

    loss_names: Tuple[str, ...]
    base_weights: Optional[Mapping[str, float]] = None
    adaptation_rate: float = 0.05
    smoothing: float = 0.95
    min_weight: float = 1e-3
    backend_hint: Optional[BackendLiteral] = None

    _weights: MutableMapping[str, float] = field(init=False, repr=False)
    _running_loss: MutableMapping[str, float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.loss_names:
            raise ValueError("loss_names must not be empty")
        if self.base_weights is not None:
            missing = [name for name in self.loss_names if name not in self.base_weights]
            if missing:
                raise ValueError(f"base_weights missing entries for: {missing}")
            weights = {name: float(self.base_weights[name]) for name in self.loss_names}
        else:
            uniform = 1.0 / len(self.loss_names)
            weights = {name: uniform for name in self.loss_names}
        total = sum(weights.values())
        if total <= 0:
            raise ValueError("base_weights must sum to a positive value")
        self._weights = {name: max(self.min_weight, weight / total) for name, weight in weights.items()}
        self._normalise_weights()
        self._running_loss = {name: 0.0 for name in self.loss_names}

    def _normalise_weights(self) -> None:
        total = sum(self._weights.values())
        if total <= 0:
            uniform = 1.0 / len(self._weights)
            for name in self._weights:
                self._weights[name] = uniform
            return
        for name in self._weights:
            self._weights[name] = max(self.min_weight, self._weights[name] / total)
        total = sum(self._weights.values())
        if total == 0:
            uniform = 1.0 / len(self._weights)
            for name in self._weights:
                self._weights[name] = uniform
        else:
            for name in self._weights:
                self._weights[name] /= total

    def update(self, losses: Mapping[str, Any]) -> Tuple[Any, Dict[str, float]]:
        """Update the adaptive weights and return the combined loss."""

        missing = [name for name in self.loss_names if name not in losses]
        if missing:
            raise KeyError(f"Missing loss values for: {missing}")

        backend = detect_backend(self.backend_hint, *losses.values())
        updated: Dict[str, float] = {}
        for name, value in losses.items():
            numeric = to_scalar(value)
            prev = self._running_loss[name]
            self._running_loss[name] = (
                self.smoothing * prev + (1.0 - self.smoothing) * numeric
            )
            baseline = max(self._running_loss[name], 1e-6)
            ratio = numeric / baseline
            inverse = 1.0 / max(ratio, 1e-6)
            current_weight = self._weights[name]
            new_weight = (1.0 - self.adaptation_rate) * current_weight + self.adaptation_rate * inverse
            updated[name] = max(self.min_weight, new_weight)

        self._weights.update(updated)
        self._normalise_weights()

        combined: Optional[Any] = None
        for name, value in losses.items():
            weighted_value = scale_value(value, self._weights[name], backend)
            if combined is None:
                combined = weighted_value
            else:
                combined = add_values(combined, weighted_value, backend)

        if combined is None:
            combined = 0.0
        return combined, dict(self._weights)

    def state_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable snapshot of the adaptor state."""

        return {
            "weights": dict(self._weights),
            "running_loss": dict(self._running_loss),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore the adaptor state from :func:`state_dict`."""

        weights = state.get("weights")
        running = state.get("running_loss")
        if not isinstance(weights, Mapping) or not isinstance(running, Mapping):
            raise ValueError("Invalid state dictionary")
        for name in self.loss_names:
            if name not in weights or name not in running:
                raise KeyError(f"State missing values for '{name}'")
        self._weights = {name: float(weights[name]) for name in self.loss_names}
        self._running_loss = {name: float(running[name]) for name in self.loss_names}
        self._normalise_weights()
