# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared fakes for the std-faster-whisper test suite.

CRITICAL: these tests NEVER instantiate a real ``WhisperModel`` or download
weights. The adapter imports ``faster_whisper`` lazily inside
``_ensure_model_loaded``; we inject a fake module via ``sys.modules`` so the
import resolves to our stub, exercising the real adapter logic against a
controllable model. Real-inference verification is a separate, opt-in script
(``scripts/verify_inference.py``), not part of this suite.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np
import pytest


class FakeWord:
    """Stand-in for a faster-whisper word with timing + probability."""

    def __init__(self, start: float, end: float, word: str, probability: float = 0.9) -> None:
        self.start = start
        self.end = end
        self.word = word
        self.probability = probability


class FakeSegment:
    """Stand-in for a faster-whisper segment."""

    def __init__(
        self,
        start: float,
        end: float,
        text: str,
        words: list[FakeWord] | None = None,
    ) -> None:
        self.start = start
        self.end = end
        self.text = text
        self.words = words
        self.temperature = 0.0
        self.avg_logprob = -0.1
        self.compression_ratio = 1.2
        self.no_speech_prob = 0.01


class FakeOptions:
    """Stand-in for ``transcription_options`` carrying whitelisted fields."""

    def __init__(self) -> None:
        self.task = "transcribe"
        self.beam_size = 5
        self.best_of = 5
        self.patience = 1.0
        self.length_penalty = 1.0
        self.repetition_penalty = 1.0
        self.no_repeat_ngram_size = 0
        self.temperatures = (0.0,)
        self.compression_ratio_threshold = 2.4
        self.log_prob_threshold = -1.0
        self.no_speech_threshold = 0.6
        self.condition_on_previous_text = True
        self.word_timestamps = False


class FakeInfo:
    """Stand-in for faster-whisper's ``TranscriptionInfo``."""

    def __init__(
        self,
        language: str | None = "en",
        *,
        with_options: bool = True,
        duration_after_vad: float | None = None,
    ) -> None:
        self.language = language
        self.language_probability = 0.97
        self.duration = 1.23
        self.transcription_options = FakeOptions() if with_options else None
        self.duration_after_vad = duration_after_vad


class FakeWhisperModel:
    """Configurable fake; records the kwargs the adapter passes to transcribe.

    A ``segments_fn`` hook lets streaming tests vary the decoded segments per
    call (e.g. by the length of the audio window passed in).
    """

    segments: list[FakeSegment] = []
    info: FakeInfo = FakeInfo()
    last_transcribe_kwargs: dict[str, Any] = {}
    last_init_kwargs: dict[str, Any] = {}
    raise_on_init: BaseException | None = None
    raise_on_transcribe: BaseException | None = None
    transcribe_calls: int = 0
    #: Optional: (audio, kwargs) -> list[FakeSegment]; overrides `segments`.
    segments_fn: Any = None

    def __init__(self, **kwargs: Any) -> None:
        if FakeWhisperModel.raise_on_init is not None:
            raise FakeWhisperModel.raise_on_init
        FakeWhisperModel.last_init_kwargs = kwargs

    def transcribe(self, source: Any, **kwargs: Any) -> tuple[list[FakeSegment], FakeInfo]:
        FakeWhisperModel.transcribe_calls += 1
        FakeWhisperModel.last_transcribe_kwargs = {"source": source, **kwargs}
        if FakeWhisperModel.raise_on_transcribe is not None:
            raise FakeWhisperModel.raise_on_transcribe
        if FakeWhisperModel.segments_fn is not None:
            return FakeWhisperModel.segments_fn(source, kwargs), FakeWhisperModel.info
        return FakeWhisperModel.segments, FakeWhisperModel.info


@pytest.fixture
def fake_faster_whisper(monkeypatch: pytest.MonkeyPatch) -> type[FakeWhisperModel]:
    """Install a fake ``faster_whisper`` module exposing ``FakeWhisperModel``.

    Returns the model class so tests can set ``segments`` / ``info`` / failure
    behaviour before invoking the adapter.
    """
    FakeWhisperModel.segments = []
    FakeWhisperModel.info = FakeInfo()
    FakeWhisperModel.last_transcribe_kwargs = {}
    FakeWhisperModel.last_init_kwargs = {}
    FakeWhisperModel.raise_on_init = None
    FakeWhisperModel.raise_on_transcribe = None
    FakeWhisperModel.transcribe_calls = 0
    FakeWhisperModel.segments_fn = None

    module = types.ModuleType("faster_whisper")
    module.WhisperModel = FakeWhisperModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    return FakeWhisperModel


def silent_pcm(seconds: float, sample_rate: int = 16000) -> bytes:
    """Return ``seconds`` of silent 16-bit LE PCM mono bytes."""
    return np.zeros(int(seconds * sample_rate), dtype="<i2").tobytes()
