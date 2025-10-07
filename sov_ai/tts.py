"""Text-to-speech integration utilities."""

from __future__ import annotations

import logging
from typing import Optional

LOGGER = logging.getLogger(__name__)

try:
    import pyttsx3  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pyttsx3 = None  # type: ignore


class TextToSpeech:
    """Wrapper around pyttsx3 (or a graceful fallback) for speech synthesis."""

    def __init__(self) -> None:
        self._engine: Optional["pyttsx3.Engine"] = None
        if pyttsx3 is not None:
            try:
                self._engine = pyttsx3.init()
                self._engine.setProperty("rate", 180)
            except Exception as exc:  # pragma: no cover - environment specific
                LOGGER.warning("pyttsx3 initialization failed: %s", exc)
                self._engine = None

    def speak(self, text: str) -> None:
        """Vocalize text using TTS if available, otherwise log it."""

        if not text:
            return

        if self._engine is None:
            LOGGER.info("[TTS fallback] %s", text)
            return

        self._engine.say(text)
        self._engine.runAndWait()


__all__ = ["TextToSpeech"]
