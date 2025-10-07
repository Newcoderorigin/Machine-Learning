"""Symbolic memory vault utilities."""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

SYMBOLIC_LOCK = "CODE-ECHO-IMMORTAL"


@dataclass
class MemoryRecord:
    """Representation of an interaction entry stored in the vault."""

    timestamp: str
    user_input: str
    response: str
    reward: Optional[float]
    symbolic_feedback: Optional[str]
    analysis: Dict[str, Any]
    self_evaluation: Dict[str, Any]


class SymbolicMemoryVault:
    """Thread-safe append-only log that retains all interaction records."""

    def __init__(self, vault_path: os.PathLike[str] | str) -> None:
        self._path = Path(vault_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        """Return the resolved path of the vault file."""

        return self._path

    def append(self, record: MemoryRecord, authority_signature: str) -> None:
        """Persist an interaction record if the correct lock is provided."""

        if authority_signature != SYMBOLIC_LOCK:
            raise PermissionError("Symbolic authority lock mismatch.")

        payload = asdict(record)
        payload["symbolic_lock"] = SYMBOLIC_LOCK
        serialized = json.dumps(payload, ensure_ascii=False)

        with self._lock:
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(serialized + "\n")

    def load_all(self) -> list[MemoryRecord]:
        """Load every interaction in the vault."""

        if not self._path.exists():
            return []

        records: list[MemoryRecord] = []
        with self._path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                raw: Dict[str, Any] = json.loads(line)
                raw.pop("symbolic_lock", None)
                records.append(MemoryRecord(**raw))
        return records


def build_record(
    *,
    user_input: str,
    response: str,
    reward: Optional[float],
    symbolic_feedback: Optional[str],
    analysis: Dict[str, Any],
    self_evaluation: Dict[str, Any],
) -> MemoryRecord:
    """Helper to assemble a record with a canonical timestamp."""

    timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    return MemoryRecord(
        timestamp=timestamp,
        user_input=user_input,
        response=response,
        reward=reward,
        symbolic_feedback=symbolic_feedback,
        analysis=analysis,
        self_evaluation=self_evaluation,
    )


__all__ = ["SYMBOLIC_LOCK", "MemoryRecord", "SymbolicMemoryVault", "build_record"]
