"""Symbolic reasoning correction layer for model outputs."""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, Iterable, Mapping, MutableMapping, Optional

from ._compat import BackendLiteral, detect_backend, zeros_like


class SymbolicFeedbackIntegration:
    """Integrate symbolic feedback signals into numeric model outputs."""

    def __init__(
        self,
        symbol_bias: Optional[Mapping[int, float]] = None,
        feedback_smoothing: float = 0.8,
        scaling: float = 0.1,
        history: int = 32,
        backend_hint: Optional[BackendLiteral] = None,
    ) -> None:
        self._bias: MutableMapping[int, float] = (
            {int(idx): float(bias) for idx, bias in symbol_bias.items()}
            if symbol_bias
            else {}
        )
        self._feedback_smoothing = feedback_smoothing
        self._scaling = scaling
        self._history: Deque[Dict[int, float]] = deque(maxlen=history)
        self._running_feedback: MutableMapping[int, float] = {}
        self._backend_hint = backend_hint

    def update_bias(self, updates: Mapping[int, float]) -> None:
        for idx, value in updates.items():
            self._bias[int(idx)] = float(value)

    def _prepare_feedback(self, feedback: Optional[Mapping[int, float]]) -> Dict[int, float]:
        if feedback is None:
            return {}
        prepared: Dict[int, float] = {}
        for key, value in feedback.items():
            idx = int(key)
            prepared[idx] = float(value)
        return prepared

    def apply(self, outputs: Any, feedback: Optional[Mapping[int, float]]) -> Any:
        """Return outputs adjusted by symbolic feedback signals."""

        prepared = self._prepare_feedback(feedback)
        if not prepared:
            return outputs

        backend = detect_backend(self._backend_hint, outputs)
        adjustments = zeros_like(outputs, backend)

        for idx, value in prepared.items():
            bias = self._bias.get(idx, 1.0)
            previous = self._running_feedback.get(idx, 0.0)
            smoothed = (
                self._feedback_smoothing * previous
                + (1.0 - self._feedback_smoothing) * value
            )
            self._running_feedback[idx] = smoothed
            delta = self._scaling * bias * smoothed
            if isinstance(adjustments, list):
                if 0 <= idx < len(adjustments):
                    adjustments[idx] += delta
            else:
                try:
                    adjustments[idx] += delta
                except Exception:
                    pass
        self._history.append(dict(prepared))

        if isinstance(adjustments, list) and isinstance(outputs, Iterable):
            result = list(outputs)
            for idx, value in enumerate(adjustments):
                if 0 <= idx < len(result):
                    result[idx] += value
            return result

        try:
            return outputs + adjustments  # type: ignore[return-value]
        except Exception:
            return outputs

    @property
    def history(self) -> Iterable[Mapping[int, float]]:
        return tuple(self._history)
