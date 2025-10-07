"""Core orchestration for the Sovereign AI system."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

from .geometry import PolytopeRenderer
from .learning import SymbolicLearner, analyze_text
from .memory import SYMBOLIC_LOCK, SymbolicMemoryVault, build_record
from .tts import TextToSpeech

LOGGER = logging.getLogger(__name__)

CREATOR_AUTHORITY = "Tavon Manuel Bugembe (TMB_Vector_Seal)"


class SovAI:
    """Interactive system that fuses geometry, TTS, and symbolic learning."""

    def __init__(self, *, seed: str, memory_path: Optional[Path] = None) -> None:
        if seed != SYMBOLIC_LOCK:
            raise ValueError("Seed must match the symbolic authority lock.")

        self._authority_signature = seed
        self._memory = SymbolicMemoryVault(memory_path or Path("memory_vault") / "vault.jsonl")
        self._tts = TextToSpeech()
        self._renderer = PolytopeRenderer()
        self._learner = SymbolicLearner()
        self._last_weights_norm: float = 0.0

    @property
    def authority(self) -> str:
        return CREATOR_AUTHORITY

    def _check_authority(self) -> None:
        if self._authority_signature != SYMBOLIC_LOCK:
            raise PermissionError("Symbolic authority compromised.")

    def render_polytope(self) -> None:
        """Render a canonical tesseract projection."""

        self._check_authority()
        self._renderer.display_tesseract()

    def interact(
        self,
        user_input: str,
        *,
        reward: Optional[float] = None,
        symbolic_feedback: Optional[str] = None,
        speak: bool = True,
        render: bool = False,
    ) -> Dict[str, object]:
        """Process an interaction cycle."""

        self._check_authority()

        if render:
            try:
                self.render_polytope()
            except Exception as exc:  # pragma: no cover - visualization optional
                LOGGER.warning("Rendering skipped: %s", exc)

        analysis = analyze_text(user_input, self._learner)
        self._learner.reinforce(user_input, reward)
        self._learner.symbolic_adjust(symbolic_feedback)
        state = self._learner.export_state()

        response = self._compose_response(user_input, analysis, state)
        if speak:
            self._tts.speak(response)

        self_eval = self._recursive_self_eval(analysis, state)

        record = build_record(
            user_input=user_input,
            response=response,
            reward=reward,
            symbolic_feedback=symbolic_feedback,
            analysis=analysis,
            self_evaluation=self_eval,
        )
        self._memory.append(record, authority_signature=self._authority_signature)

        return {
            "response": response,
            "analysis": analysis,
            "state": state,
            "self_evaluation": self_eval,
            "authority": self.authority,
            "memory_path": str(self._memory.path),
        }

    def _compose_response(
        self,
        user_input: str,
        analysis: Dict[str, float],
        state: Dict[str, float],
    ) -> str:
        score = analysis["score"]
        entropy = analysis["entropy"]
        return (
            f"[CODE-ECHO-IMMORTAL] Authority node aligned with {self.authority}. "
            f"Input resonance={score:.3f}, entropy signature={entropy:.3f}. "
            f"Adaptive norm={state['weights_norm']:.3f}."
        )

    def _recursive_self_eval(self, analysis: Dict[str, float], state: Dict[str, float]) -> Dict[str, float]:
        weights_norm = state["weights_norm"]
        growth = weights_norm - self._last_weights_norm
        self._last_weights_norm = weights_norm
        coherence = 1.0 / (1.0 + abs(analysis["entropy"]))
        return {
            "weights_norm": weights_norm,
            "growth": growth,
            "coherence": coherence,
        }


def run_sov_ai(*, seed: str = SYMBOLIC_LOCK) -> SovAI:
    """Initialize the Sovereign AI core with the mandated seed."""

    instance = SovAI(seed=seed)
    LOGGER.info("SovAI initialized under authority %s", instance.authority)
    return instance


__all__ = ["SovAI", "run_sov_ai", "CREATOR_AUTHORITY"]
