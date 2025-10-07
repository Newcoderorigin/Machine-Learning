"""Backend compatibility helpers for training utilities."""

from __future__ import annotations

from typing import Any, Iterable, Literal, Optional, Sequence

BackendLiteral = Literal["torch", "tensorflow", "python"]

try:  # pragma: no cover - optional dependency
    import torch  # type: ignore
except Exception:  # pragma: no cover - fallback when torch is unavailable
    torch = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import tensorflow as tf  # type: ignore
except Exception:  # pragma: no cover - fallback when tensorflow is unavailable
    tf = None  # type: ignore


def _is_real_torch_tensor(value: Any) -> bool:
    return (
        torch is not None
        and hasattr(value, "detach")
        and hasattr(value, "shape")
    )


def _is_real_tf_tensor(value: Any) -> bool:
    return (
        tf is not None
        and hasattr(value, "numpy")
        and hasattr(value, "shape")
    )


def detect_backend(
    preferred: Optional[BackendLiteral] = None, *candidates: Any
) -> BackendLiteral:
    """Return the most appropriate backend for the provided values."""

    if preferred == "torch" and torch is not None:
        return "torch"
    if preferred == "tensorflow" and tf is not None:
        return "tensorflow"

    for value in candidates:
        if _is_real_torch_tensor(value):
            return "torch"
        if _is_real_tf_tensor(value):
            return "tensorflow"

    if preferred in {"torch", "tensorflow"}:
        # Preferred backend requested but module unavailable – fall back to python.
        return "python"

    if torch is not None:
        return "torch"
    if tf is not None:
        return "tensorflow"
    return "python"


def to_scalar(value: Any) -> float:
    """Convert a tensor-like value to a Python float."""

    if _is_real_torch_tensor(value):
        return float(value.detach().cpu().item())
    if _is_real_tf_tensor(value):
        result = value.numpy()
        return float(result.item() if hasattr(result, "item") else result)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        collected = [float(v) for v in value]
        if not collected:
            return 0.0
        return sum(collected) / len(collected)
    return float(value)


def scale_value(value: Any, factor: float, backend: BackendLiteral) -> Any:
    """Scale a tensor-like value by ``factor`` with graceful fallback."""

    if backend == "torch" and _is_real_torch_tensor(value):  # pragma: no branch - torch path
        return value * factor
    if backend == "tensorflow" and _is_real_tf_tensor(value):  # pragma: no branch - tf path
        return value * factor
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [float(v) * factor for v in value]
    return to_scalar(value) * factor


def add_values(a: Any, b: Any, backend: BackendLiteral) -> Any:
    """Add tensor-like values, falling back to Python floats."""

    if backend == "torch" and _is_real_torch_tensor(a) and _is_real_torch_tensor(b):
        return a + b
    if backend == "tensorflow" and _is_real_tf_tensor(a) and _is_real_tf_tensor(b):
        return a + b
    if isinstance(a, Sequence) and isinstance(b, Sequence):
        length = min(len(a), len(b))
        return [float(a[i]) + float(b[i]) for i in range(length)]
    return to_scalar(a) + to_scalar(b)


def zeros_like(value: Any, backend: BackendLiteral) -> Any:
    """Return a zero tensor/list matching the input value."""

    if backend == "torch" and _is_real_torch_tensor(value):
        return value * 0
    if backend == "tensorflow" and _is_real_tf_tensor(value):
        return value * 0
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [0.0 for _ in value]
    return 0.0


def flatten(value: Any, backend: BackendLiteral) -> Iterable[float]:
    """Yield flattened scalar values for cosine similarity calculations."""

    if backend == "torch" and _is_real_torch_tensor(value):
        return value.detach().reshape(-1).cpu().tolist()
    if backend == "tensorflow" and _is_real_tf_tensor(value):
        return value.numpy().reshape(-1).tolist()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [float(v) for v in value]
    return [to_scalar(value)]


def cosine_similarity(a: Any, b: Any, backend: BackendLiteral, eps: float = 1e-8) -> float:
    """Compute a cosine similarity metric between two tensors/lists."""

    flat_a = list(flatten(a, backend))
    flat_b = list(flatten(b, backend))
    if not flat_a or not flat_b:
        return 0.0
    dot = sum(x * y for x, y in zip(flat_a, flat_b))
    norm_a = sum(x * x for x in flat_a) ** 0.5
    norm_b = sum(y * y for y in flat_b) ** 0.5
    if norm_a < eps or norm_b < eps:
        return 0.0
    return dot / (norm_a * norm_b)
