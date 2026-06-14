# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the pure conversion / config helpers."""

from __future__ import annotations

import numpy as np
import pytest

from std_faster_whisper._config import FasterWhisperParams, provider_kwargs
from std_faster_whisper._convert import (
    convert_segments,
    pcm_s16le_to_float32,
    safe_extra,
)

from .conftest import FakeInfo, FakeSegment, FakeWord


# --------------------------------------------------------------------------- #
# PCM decode
# --------------------------------------------------------------------------- #
def test_pcm_decode_roundtrip() -> None:
    samples = np.array([0, 16384, -16384, 32767, -32768], dtype="<i2")
    decoded = pcm_s16le_to_float32(samples.tobytes())
    assert decoded.dtype == np.float32
    assert decoded[0] == pytest.approx(0.0)
    assert decoded[1] == pytest.approx(0.5, abs=1e-4)
    assert decoded[3] == pytest.approx(1.0, abs=1e-4)


def test_pcm_decode_empty() -> None:
    assert pcm_s16le_to_float32(b"").size == 0
    assert pcm_s16le_to_float32(b"\x00").size == 0  # single odd byte


def test_pcm_decode_drops_trailing_odd_byte() -> None:
    # 5 bytes = 2 whole samples + 1 dangling byte.
    decoded = pcm_s16le_to_float32(b"\x00\x00\x00\x40\x7f")
    assert decoded.size == 2


# --------------------------------------------------------------------------- #
# Segment conversion
# --------------------------------------------------------------------------- #
def test_convert_segments_with_and_without_words() -> None:
    segs = [
        FakeSegment(0.0, 1.0, "a", words=[FakeWord(0.0, 0.5, "a", 0.9)]),
        FakeSegment(1.0, 2.0, "b", words=None),
    ]
    segments, words = convert_segments(segs)
    assert len(segments) == 2
    assert [w.text for w in words] == ["a"]
    assert segments[0].words is not None
    assert segments[1].words is None


# --------------------------------------------------------------------------- #
# provider_kwargs
# --------------------------------------------------------------------------- #
def test_provider_kwargs_none_returns_empty() -> None:
    assert provider_kwargs(None) == {}


def test_provider_kwargs_omits_temperature_when_none() -> None:
    kwargs = provider_kwargs(FasterWhisperParams(temperature=None))
    assert "temperature" not in kwargs
    assert kwargs["beam_size"] == 5


def test_provider_kwargs_includes_temperature_when_set() -> None:
    kwargs = provider_kwargs(FasterWhisperParams(temperature=0.4))
    assert kwargs["temperature"] == 0.4


def test_provider_kwargs_foreign_type_falls_back_to_defaults() -> None:
    # Defensive: a non-FasterWhisperParams instance falls back to defaults rather
    # than raising (the real swap-safety raise happens earlier, in the gate).
    from standard_asr.runtime_params import ProviderParams

    class Other(ProviderParams):
        x: int = 1

    kwargs = provider_kwargs(Other())
    assert kwargs["beam_size"] == 5
    assert kwargs["task"] == "transcribe"


# --------------------------------------------------------------------------- #
# safe_extra
# --------------------------------------------------------------------------- #
def test_safe_extra_whitelists_options() -> None:
    extra = safe_extra(FakeInfo(duration_after_vad=0.9))
    opts = extra["transcription_options"]
    assert opts["task"] == "transcribe"
    assert "initial_prompt" not in opts  # never echoed back
    assert extra["duration_after_vad"] == pytest.approx(0.9)


def test_safe_extra_without_options_or_vad() -> None:
    extra = safe_extra(FakeInfo(with_options=False, duration_after_vad=None))
    assert extra["transcription_options"] == {}
    assert "duration_after_vad" not in extra


def test_safe_extra_skips_absent_whitelisted_fields() -> None:
    class _PartialOptions:
        beam_size = 7

    class _Info:
        transcription_options = _PartialOptions()
        duration_after_vad = None

    extra = safe_extra(_Info())
    assert extra["transcription_options"] == {"beam_size": 7}
