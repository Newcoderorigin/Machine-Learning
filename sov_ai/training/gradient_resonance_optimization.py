"""Gradient modulation utilities based on inter-layer coherence."""

from __future__ import annotations

from typing import Any, Dict, Mapping, MutableMapping, Optional

from ._compat import (
    BackendLiteral,
    add_values,
    cosine_similarity,
    detect_backend,
    scale_value,
)


class GradientResonanceOptimization:
    """Modulate gradients according to inter-layer resonance signals."""

    def __init__(
        self,
        resonance_strength: float = 0.25,
        smoothing: float = 0.9,
        min_scale: float = 0.5,
        max_scale: float = 1.5,
        backend_hint: Optional[BackendLiteral] = None,
    ) -> None:
        self._resonance_strength = resonance_strength
        self._smoothing = smoothing
        self._min_scale = min_scale
        self._max_scale = max_scale
        self._backend_hint = backend_hint
        self._running_signature: MutableMapping[str, Any] = {}

    def _update_signature(self, name: str, gradient: Any, backend: BackendLiteral) -> float:
        previous = self._running_signature.get(name)
        if previous is None:
            self._running_signature[name] = gradient
            return 0.0
        coherence = cosine_similarity(previous, gradient, backend)
        blended = scale_value(previous, self._smoothing, backend)
        residual = scale_value(gradient, 1.0 - self._smoothing, backend)
        try:
            updated = add_values(blended, residual, backend)
        except Exception:
            updated = residual
        self._running_signature[name] = updated
        return float(max(-1.0, min(1.0, coherence)))

    def modulate(self, gradients: Mapping[str, Any]) -> Dict[str, Any]:
        if not gradients:
            return {}
        backend = detect_backend(self._backend_hint, *gradients.values())
        scaled: Dict[str, Any] = {}
        for name, grad in gradients.items():
            coherence = self._update_signature(name, grad, backend)
            scale = 1.0 + self._resonance_strength * coherence
            scale = max(self._min_scale, min(self._max_scale, scale))
            scaled[name] = scale_value(grad, scale, backend)
        return scaled

    def apply(self, gradients: Mapping[str, Any], optimizer: Any) -> Dict[str, Any]:
        """Apply modulated gradients and step a PyTorch-style optimizer."""

        scaled = self.modulate(gradients)
        param_groups = getattr(optimizer, "param_groups", None)
        if param_groups is None:
            raise TypeError("Optimizer does not expose param_groups for modulation")
        for group in param_groups:
            for param in group.get("params", []):
                target = None
                if param in scaled:
                    target = scaled[param]
                else:
                    name = getattr(param, "name", None)
                    if name and name in scaled:
                        target = scaled[name]
                if target is None:
                    continue
                setattr(param, "grad", target)
        optimizer.step()
        return scaled

    def reset(self) -> None:
        self._running_signature.clear()
