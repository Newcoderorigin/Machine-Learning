"""Automated drift and overfitting auditing for training loops."""

from __future__ import annotations

from collections import deque
from statistics import mean
from typing import Deque, Dict, Iterable, Mapping, MutableMapping


class RecursiveSelfAuditMechanism:
    """Monitor metric histories and flag drift/overfitting conditions."""

    def __init__(
        self,
        window: int = 50,
        drift_threshold: float = 0.1,
        overfit_threshold: float = 0.05,
    ) -> None:
        if window <= 1:
            raise ValueError("window must be greater than 1")
        self._window = window
        self._drift_threshold = drift_threshold
        self._overfit_threshold = overfit_threshold
        self._metrics: MutableMapping[str, Deque[float]] = {}

    def record(self, metrics: Mapping[str, float]) -> None:
        """Record a new set of metrics into the auditing buffers."""

        for name, value in metrics.items():
            history = self._metrics.setdefault(name, deque(maxlen=self._window))
            history.append(float(value))

    def _rolling_mean(self, values: Iterable[float]) -> float:
        collected = list(values)
        return mean(collected) if collected else 0.0

    def audit(self) -> Dict[str, bool]:
        """Return drift and overfitting alerts based on recorded metrics."""

        if not self._metrics:
            return {"drift_alert": False, "overfit_alert": False}

        drift_alert = False
        overfit_alert = False

        for name, values in self._metrics.items():
            if len(values) < 2:
                continue
            midpoint = max(1, len(values) // 2)
            early = self._rolling_mean(list(values)[:midpoint])
            recent = self._rolling_mean(list(values)[midpoint:])
            if early == 0:
                continue
            change = abs(recent - early) / max(abs(early), 1e-6)
            if change >= self._drift_threshold:
                drift_alert = True
            if name.startswith("train_"):
                companion = name.replace("train_", "val_", 1)
                if companion in self._metrics:
                    val_recent = self._rolling_mean(list(self._metrics[companion])[midpoint:])
                    gap = val_recent - recent
                    if gap >= self._overfit_threshold:
                        overfit_alert = True

        return {"drift_alert": drift_alert, "overfit_alert": overfit_alert}

    def state_dict(self) -> Dict[str, Iterable[float]]:
        return {name: tuple(history) for name, history in self._metrics.items()}

    def load_state_dict(self, state: Mapping[str, Iterable[float]]) -> None:
        self._metrics.clear()
        for name, values in state.items():
            history = deque(maxlen=self._window)
            for value in values:
                history.append(float(value))
            self._metrics[name] = history
