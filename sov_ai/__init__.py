"""Sovereign AI interactive system package."""

from importlib import import_module
from typing import Any

__all__ = ["SovAI", "run_sov_ai", "training"]


def __getattr__(name: str) -> Any:  # pragma: no cover - import side-effect
    if name in {"SovAI", "run_sov_ai"}:
        core = import_module("sov_ai.core")
        globals()["SovAI"] = core.SovAI
        globals()["run_sov_ai"] = core.run_sov_ai
        return globals()[name]
    if name == "training":
        module = import_module("sov_ai.training")
        globals()[name] = module
        return module
    raise AttributeError(f"module 'sov_ai' has no attribute '{name}'")
