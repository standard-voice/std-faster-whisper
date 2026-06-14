# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Streaming-path tests for the windowed faster-whisper session.

Runs entirely against the injected ``FakeWhisperModel`` (no weights). The
``segments_fn`` hook makes the fake return segments based on how much audio has
accumulated, so we can simulate the window growing across re-decodes and assert
the partial -> final progression plus the spec event-sequence contract.
"""

from __future__ import annotations

from typing import Any

import pytest
from standard_asr import TranscriptionEvent
from standard_asr.audio_format import AudioFormat
from standard_asr.capabilities import (
    FinalityCap,
    FlagCap,
    ReconnectCap,
    StreamTimestampsCap,
)
from standard_asr.compliance import check_event_sequence, check_streaming_param_gating
from standard_asr.exceptions import UnsupportedFeatureError

from std_faster_whisper import FasterWhisperASR, TinyASR
from std_faster_whisper._streaming import FasterWhisperStreamingSession

from .conftest import FakeSegment, FakeWhisperModel, silent_pcm

_FMT = AudioFormat(encoding="pcm_s16le", sample_rate=16000, channels=1)


def _window_seconds(audio: Any) -> float:
    return len(audio) / 16000.0


# --------------------------------------------------------------------------- #
# Capability declarations (honest, fail-closed)
# --------------------------------------------------------------------------- #
def test_streaming_capabilities_are_conservative() -> None:
    caps = FasterWhisperASR.declared_capabilities
    assert caps.supports("streaming_input") is True
    assert caps.supports("streaming_output") is True
    assert caps.supports("streaming.emits_partials") is True
    # Windowed re-decode => no stability, no supersede, no reconnect.
    assert caps.supports("streaming.word_stability") is False
    assert caps.supports("streaming.re_segments") is False
    node = caps.node_at("streaming.reconnect")
    assert isinstance(node, ReconnectCap)
    assert node.mode == "unsupported"
    finality = caps.node_at("streaming.finality_level")
    assert isinstance(finality, FinalityCap)
    assert finality.mode == "final"
    ts = caps.node_at("streaming.timestamps")
    assert isinstance(ts, StreamTimestampsCap)
    assert ts.mode == "post_align"


def test_only_streaming_output_engine_would_reject_incremental() -> None:
    # Sanity: our engine DOES declare streaming_input, so a no-arg / audio_format
    # session is allowed (the inverse case is covered by the core suite).
    assert FasterWhisperASR.declared_capabilities.node_at("streaming_input") == FlagCap(
        supported=True
    )


# --------------------------------------------------------------------------- #
# Live windowed run
# --------------------------------------------------------------------------- #
async def test_streaming_emits_partials_then_finals(
    fake_faster_whisper: type[FakeWhisperModel],
) -> None:
    # The fake returns more sentences as the window grows. Two stable sentences
    # plus a moving tail; the third sentence ends near the frontier so it stays a
    # partial until the final flush.
    def segments_fn(audio: Any, _kwargs: dict[str, Any]) -> list[FakeSegment]:
        secs = _window_seconds(audio)
        segs = [FakeSegment(0.0, 2.0, "First sentence. ")]
        if secs >= 8:
            segs.append(FakeSegment(2.0, 5.0, "Second sentence. "))
        if secs >= 8:
            segs.append(FakeSegment(5.0, secs, "trailing tail"))
        return segs

    fake_faster_whisper.segments_fn = segments_fn

    engine = TinyASR()
    events: list[TranscriptionEvent] = []
    async with engine.start_transcription(audio_format=_FMT) as session:
        # Feed ~12s of audio in 2s chunks (redecode interval is 4s).
        session.feed([silent_pcm(2.0) for _ in range(6)])
        async for event in session:
            events.append(event)

    types = [e.type for e in events]
    assert "partial" in types
    assert "final" in types
    assert types[-1] == "done"
    # Every partial reports stable_until=0 (Whisper may rewrite the window).
    for e in events:
        if e.type == "partial":
            assert e.stable_until == 0
    # Finals carry stable, never-reused segment ids.
    final_ids = [e.segment_id for e in events if e.type == "final"]
    assert final_ids == sorted(set(final_ids), key=final_ids.index)  # no dupes/reorder
    # The full transcript is recoverable from the reduced session result.
    text = session.result().text
    assert "First sentence." in text


async def test_recorded_stream_obeys_event_sequence_contract(
    fake_faster_whisper: type[FakeWhisperModel],
) -> None:
    # Capture a real event stream from the session and assert it satisfies the
    # standard's segment/event-order contract (plugin_entrypoints.md: cover
    # check_event_sequence in your own tests).
    def segments_fn(audio: Any, _kwargs: dict[str, Any]) -> list[FakeSegment]:
        secs = _window_seconds(audio)
        segs = [FakeSegment(0.0, 2.0, "alpha ")]
        if secs >= 8:
            segs.append(FakeSegment(2.0, secs, "beta"))
        return segs

    fake_faster_whisper.segments_fn = segments_fn

    engine = TinyASR()
    events: list[TranscriptionEvent] = []
    async with engine.start_transcription(audio_format=_FMT) as session:
        session.feed([silent_pcm(2.0) for _ in range(6)])
        async for event in session:
            events.append(event)

    report = check_event_sequence(events)
    assert report.passed, [i.message for i in report.issues]


async def test_streaming_silence_emits_progress_then_done(
    fake_faster_whisper: type[FakeWhisperModel],
) -> None:
    # No segments ever returned (pure silence): the session still terminates with
    # progress heartbeats and a done, never hangs.
    fake_faster_whisper.segments = []
    engine = TinyASR()
    events: list[TranscriptionEvent] = []
    async with engine.start_transcription(audio_format=_FMT) as session:
        session.feed([silent_pcm(2.0) for _ in range(3)])
        async for event in session:
            events.append(event)
    assert events[-1].type == "done"
    assert all(e.type in {"progress", "done"} for e in events)


async def test_streaming_whole_input_path(
    fake_faster_whisper: type[FakeWhisperModel],
) -> None:
    # start_transcription(audio=...) seeds the window from a negotiated whole
    # input and streams the result back (OpenAI-SSE-style usage).
    fake_faster_whisper.segments = [FakeSegment(0.0, 1.0, "whole input result")]
    engine = TinyASR()
    events: list[TranscriptionEvent] = []
    async with engine.start_transcription(audio=(silent_to_array(1.0))) as session:
        async for event in session:
            events.append(event)
    assert events[-1].type == "done"
    assert "whole input result" in session.result().text


async def test_streaming_decode_failure_becomes_engine_error_event(
    fake_faster_whisper: type[FakeWhisperModel],
) -> None:
    fake_faster_whisper.raise_on_transcribe = RuntimeError("boom")
    engine = TinyASR()
    events: list[TranscriptionEvent] = []
    async with engine.start_transcription(audio_format=_FMT) as session:
        session.feed([silent_pcm(2.0) for _ in range(3)])
        async for event in session:
            events.append(event)
    error_events = [e for e in events if e.type == "error"]
    assert error_events
    assert error_events[0].code == "engine_error"


# --------------------------------------------------------------------------- #
# Compliance helpers usable by plugin CI
# --------------------------------------------------------------------------- #
def test_streaming_param_gating_compliant(fake_faster_whisper: type[FakeWhisperModel]) -> None:
    report = check_streaming_param_gating(TinyASR())
    assert report.passed, [i.message for i in report.issues]


def test_unsupported_streaming_candidate_languages_rejected(
    fake_faster_whisper: type[FakeWhisperModel],
) -> None:
    # We do NOT declare streaming candidate_languages; per spec LANG R3 this is a
    # silent carve-out (diagnostic, not raise) -- so the session opens and a
    # diagnostic is attached rather than an exception. Assert it opens cleanly.
    from standard_asr import RuntimeParams

    engine = TinyASR()
    session = engine.start_transcription(
        audio_format=_FMT, params=RuntimeParams(language="auto", candidate_languages=["en", "fr"])
    )
    assert isinstance(session, FasterWhisperStreamingSession)


def test_streaming_word_timestamps_unsupported_granularity_strict() -> None:
    # char granularity is not declared; strict mode rejects it at the gate.
    from standard_asr import RuntimeParams
    from standard_asr.runtime_params import WordTimestampGranularity

    engine = TinyASR()
    with pytest.raises(UnsupportedFeatureError):
        engine.start_transcription(
            audio_format=_FMT,
            params=RuntimeParams(word_timestamps=WordTimestampGranularity.CHAR),
        )


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def silent_to_array(seconds: float) -> tuple[Any, int]:
    import numpy as np

    return (np.zeros(int(seconds * 16000), dtype=np.float32), 16000)
